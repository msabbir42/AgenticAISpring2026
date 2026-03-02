import os
import sqlite3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

# =============================================================================
# TRACING & DEVICE CONFIGURATION
# =============================================================================

def decide_whether_to_print(env_var_name: str, default_value: bool = False) -> bool:
    env_value = os.getenv(env_var_name)
    if env_value is None: return default_value
    normalized_value = env_value.strip().lower()
    if normalized_value in {"verbose", "true", "1", "yes"}: return True
    if normalized_value in {"quiet", "false", "0", "no"}: return False
    return default_value

initial_trace_enabled = decide_whether_to_print("bool_verbose", default_value=True)

def get_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(MessagesState):
    should_exit: bool
    trace_enabled: bool
    awaiting_llm: bool
    active_model: str  

# =============================================================================
# MODEL CREATION
# =============================================================================

def create_llm(model_id: str) -> dict:
    device = get_device()
    print(f"Loading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    if device in {"cuda", "mps"}:
        model = model.to(device)

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return {"llm": llm, "tokenizer": tokenizer}

# =============================================================================
# MESSAGE / PROMPT HELPERS
# =============================================================================

def build_prompt_from_messages(messages: list, active_model: str, tokenizer) -> str:
    chat_messages = []
    other_model = "Qwen" if active_model == "Llama" else "Llama"
    
    sys_content = f"You are {active_model}. You are in a chat with a Human and another AI named {other_model}."
    chat_messages.append({"role": "system", "content": sys_content})

    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue 
            
        if isinstance(msg, HumanMessage):
            chat_messages.append({"role": "user", "content": f"Human: {msg.content}"})
        elif isinstance(msg, AIMessage):
            speaker = msg.name if msg.name else "Assistant"
            if speaker == active_model:
                chat_messages.append({"role": "assistant", "content": f"{speaker}: {msg.content}"})
            else:
                chat_messages.append({"role": "user", "content": f"{speaker}: {msg.content}"})

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    transcript_lines = [f"System: {sys_content}"]
    for m in chat_messages[1:]:
        transcript_lines.append(f"{m['content']}")
    transcript_lines.append(f"{active_model}:")
    return "\n".join(transcript_lines)

def clean_model_response(raw_response: str) -> str:
    cleaned = raw_response.strip()
    for stop_token in ["\nUser:", "\nHuman:", "<|eot_id|>", "<|im_end|>"]:
        if stop_token in cleaned:
            cleaned = cleaned.split(stop_token, 1)[0]
    return cleaned.strip()

# =============================================================================
# GRAPH CREATION
# =============================================================================

def create_graph(llama_bundle: dict, qwen_bundle: dict, checkpointer):

    def get_user_input(state: AgentState) -> dict:
        print("\n" + "=" * 50)
        print("Enter your text ('verbose', 'quiet', or 'quit' to exit):")
        print("=" * 50)
        print("\n> ", end="")
        
        user_input = input()
        print("Input: ", user_input)
        normalized_input = user_input.strip().lower()

        if normalized_input in {"quit", "exit", "q"}:
            return {"should_exit": True, "awaiting_llm": False}
        if normalized_input == "verbose":
            print("Tracing enabled.")
            return {"should_exit": False, "trace_enabled": True, "awaiting_llm": False}
        if normalized_input == "quiet":
            return {"should_exit": False, "trace_enabled": False, "awaiting_llm": False}
        if normalized_input == "":
            return {"should_exit": False, "awaiting_llm": False}

        if "hey qwen" in normalized_input:
            active_model = "Qwen"
        else:
            active_model = "Llama"

        return {
            "messages": [HumanMessage(content=user_input)],
            "should_exit": False,
            "awaiting_llm": True,
            "active_model": active_model
        }

    def call_model(state: AgentState) -> dict:
        active = state.get("active_model", "Llama")
        bundle = qwen_bundle if active == "Qwen" else llama_bundle

        if state.get("trace_enabled", False):
            print(f"[TRACE] Routing to {active}...")

        prompt_text = build_prompt_from_messages(
            state["messages"],
            active,
            bundle.get("tokenizer"),
        )
        
        raw_response = bundle["llm"].invoke(prompt_text)
        cleaned_response = clean_model_response(raw_response)

        return {
            "messages": [AIMessage(content=cleaned_response, name=active)],
            "awaiting_llm": False,
        }

    def print_latest_response(state: AgentState) -> dict:
        latest_message = state["messages"][-1]
        speaker = latest_message.name if latest_message.name else "Assistant"
        
        print("\n" + "-" * 50)
        print(f"{speaker} response")
        print("-" * 50)
        print(str(latest_message.content))
        return {}

    def route_after_input(state: AgentState) -> str:
        if state.get("should_exit", False): return END
        if not state.get("awaiting_llm", False): return "get_user_input"
        return "call_model"

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_model", call_model)
    graph_builder.add_node("print_latest_response", print_latest_response)

    graph_builder.add_edge(START, "get_user_input")
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {"call_model": "call_model", "get_user_input": "get_user_input", END: END}
    )
    graph_builder.add_edge("call_model", "print_latest_response")
    graph_builder.add_edge("print_latest_response", "get_user_input")

    # Compile the graph using the provided SQLite checkpointer
    return graph_builder.compile(checkpointer=checkpointer)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    print("=" * 50)
    print("LangGraph Multi-Agent with Crash Recovery")
    print("=" * 50)

    llama_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    qwen_model_id = "Qwen/Qwen2.5-0.5B"

    llama_bundle = create_llm(llama_model_id)
    qwen_bundle = create_llm(qwen_model_id)

    # Use a context manager for safe SQLite handling
    with SqliteSaver.from_conn_string("chat_history.db") as memory_saver:
        
        graph = create_graph(llama_bundle, qwen_bundle, memory_saver)
        thread_config = {"configurable": {"thread_id": "session_1"}}
        
        state_snapshot = graph.get_state(thread_config)

        if not state_snapshot.values:
            print("\n[INFO] Starting a new conversation thread...")
            initial_state = {
                "messages": [],
                "should_exit": False,
                "trace_enabled": initial_trace_enabled,
                "awaiting_llm": False,
                "active_model": "Llama" 
            }
            graph.invoke(initial_state, config=thread_config)
        else:
            print("\n[INFO] Crash recovery successful! Resuming your previous conversation...")
            
            # Extract and print the existing messages to prove recovery
            recovered_messages = state_snapshot.values.get("messages", [])
            print("\n" + "~" * 50)
            print("RECOVERED CHAT HISTORY")
            print("~" * 50)
            for msg in recovered_messages:
                if isinstance(msg, HumanMessage):
                    print(f"Human: {msg.content}")
                elif isinstance(msg, AIMessage):
                    speaker = msg.name if msg.name else "Assistant"
                    print(f"{speaker}: {msg.content}")
            print("~" * 50 + "\n")

            # CRITICAL FIX: Determine how to resume based on the pending queue
            if state_snapshot.next:
                # If there is a pending node (e.g., ('get_user_input',)), resume it directly
                graph.invoke(None, config=thread_config)
            else:
                # If the pending queue was cleared by the interrupt, trigger a new 
                # invocation from the START node using an empty state update.
                graph.invoke({"awaiting_llm": False}, config=thread_config)

if __name__ == "__main__":
    main()