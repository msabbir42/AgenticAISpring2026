# langgraph_simple_llama_agent.py
# Program demonstrates use of LangGraph for a simple single-model chat agent.
# It writes to stdout and asks the user to enter a line of text through stdin.
# It supports three special commands:
#   - verbose: enable tracing information
#   - quiet: disable tracing information
#   - quit / exit / q: terminate the graph
# Empty input is never sent to the model. Instead, the graph routes back to
# the input node and asks again.
# Normal input is sent to Llama only.
# The program maintains chat history using LangGraph's Message API.
# Qwen routing has been disabled for this version.
# The program uses CUDA if available, then MPS if available, otherwise CPU.
# After the LangGraph graph is created but before it executes, the program
# uses Mermaid to write an image of the graph to the file lg_graph.png.

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph


# =============================================================================
# TRACING CONFIGURATION
# =============================================================================

def decide_whether_to_print(env_var_name: str, default_value: bool = False) -> bool:
    """
    Read an environment variable and convert it into a boolean flag.

    Accepted truthy values: verbose, true, 1, yes
    Accepted falsy values: quiet, false, 0, no

    If the variable is not set, return the provided default value.
    If the variable is set to an unrecognized string, also return the default.
    """
    env_value = os.getenv(env_var_name)

    if env_value is None:
        return default_value

    normalized_value = env_value.strip().lower()

    if normalized_value in {"verbose", "true", "1", "yes"}:
        return True

    if normalized_value in {"quiet", "false", "0", "no"}:
        return False

    return default_value


initial_trace_enabled = decide_whether_to_print("bool_verbose", default_value=True)


# =============================================================================
# DEVICE SELECTION
# =============================================================================

def get_device() -> str:
    """
    Detect and return the best available compute device.

    Returns:
        - "cuda" for NVIDIA GPUs
        - "mps" for Apple Silicon
        - "cpu" otherwise
    """
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"

    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"

    print("Using CPU for inference")
    return "cpu"


# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(MessagesState):
    """
    State object that flows through the LangGraph nodes.

    Fields:
    - messages: conversation history stored using the LangGraph Message API
    - should_exit: Boolean flag indicating whether the graph should terminate
    - trace_enabled: Whether tracing information should be printed
    - awaiting_llm: Whether a new human message was just added and should be
      routed to the LLM

    Graph flow:
        START -> get_user_input -> [conditional]
                                  |-> END
                                  |-> get_user_input            (for empty input or mode commands)
                                  |-> call_llama -> print_latest_response -> get_user_input
    """

    should_exit: bool
    trace_enabled: bool
    awaiting_llm: bool


# =============================================================================
# MODEL CREATION
# =============================================================================

def create_llm(model_id: str) -> dict:
    """
    Create and configure an instruction-tuned causal language model from
    Hugging Face and return both the wrapped LangChain model and tokenizer.
    """
    device = get_device()

    print(f"Loading model: {model_id}")
    print("This may take a moment on first run as the model is downloaded...")

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

    print("Model loaded successfully!")
    return {"llm": llm, "tokenizer": tokenizer}


# =============================================================================
# MESSAGE / PROMPT HELPERS
# =============================================================================

def message_to_role_and_content(message) -> tuple[str, str]:
    """
    Convert a LangChain message object into a role/content pair that can be
    passed to a chat template.
    """
    if isinstance(message, SystemMessage):
        return "system", str(message.content)

    if isinstance(message, HumanMessage):
        return "user", str(message.content)

    if isinstance(message, AIMessage):
        return "assistant", str(message.content)

    if isinstance(message, ToolMessage):
        return "tool", str(message.content)

    if isinstance(message, FunctionMessage):
        return "function", str(message.content)

    return "user", str(message.content)


def build_prompt_from_messages(messages: list, tokenizer) -> str:
    """
    Build a model prompt from LangChain messages.

    If the tokenizer provides a chat template, use it. Otherwise, fall back to a
    simple text transcript.
    """
    chat_messages = []

    for message in messages:
        role_name, content_text = message_to_role_and_content(message)
        chat_messages.append({"role": role_name, "content": content_text})

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    transcript_lines = []

    for chat_message in chat_messages:
        role_name = chat_message["role"]
        content_text = chat_message["content"]

        if role_name == "system":
            transcript_lines.append(f"System: {content_text}")
        elif role_name == "assistant":
            transcript_lines.append(f"Assistant: {content_text}")
        elif role_name == "tool":
            transcript_lines.append(f"Tool: {content_text}")
        elif role_name == "function":
            transcript_lines.append(f"Function: {content_text}")
        else:
            transcript_lines.append(f"User: {content_text}")

    transcript_lines.append("Assistant:")
    return "\n".join(transcript_lines)


def clean_model_response(raw_response: str) -> str:
    """
    Clean a model response so only the assistant text is kept.
    """
    cleaned_response = raw_response.strip()

    if "\nUser:" in cleaned_response:
        cleaned_response = cleaned_response.split("\nUser:", 1)[0]

    if "<|eot_id|>" in cleaned_response:
        cleaned_response = cleaned_response.split("<|eot_id|>", 1)[0]
    return cleaned_response.strip()


# =============================================================================
# GRAPH CREATION
# =============================================================================

def create_graph(llama_bundle: dict):
    """
    Create the LangGraph state graph with the following nodes:
    1. get_user_input: Reads input from stdin and updates state
    2. call_llama: Sends the full message history to the Llama model
    3. print_latest_response: Prints the latest AI response

    The graph uses a conditional edge after get_user_input so that:
    - quit / exit / q -> END
    - empty input -> get_user_input
    - verbose / quiet -> get_user_input
    - all other normal input -> call_llama
    """

    def get_user_input(state: AgentState) -> dict:
        """
        Prompt the user for input and update the relevant state fields.

        Special commands:
        - quit / exit / q: terminate the graph
        - verbose: enable tracing, then ask for another input
        - quiet: disable tracing, then ask for another input
        - empty input: ask again without calling the model
        - otherwise: append a HumanMessage and route to the model
        """
        trace_enabled = state.get("trace_enabled", False)

        if trace_enabled:
            print("[TRACE] Entering node: get_user_input")

        print("\n" + "=" * 50)
        print("Enter your text ('verbose', 'quiet', or 'quit' to exit):")
        print("=" * 50)

        print("\n> ", end="")
        user_input = input()
        normalized_input = user_input.strip().lower()

        if normalized_input in {"quit", "exit", "q"}:
            if trace_enabled:
                print("[TRACE] Quit command detected. Routing to END.")
            return {
                "should_exit": True,
                "awaiting_llm": False,
            }

        if normalized_input == "verbose":
            print("Tracing enabled.")
            return {
                "should_exit": False,
                "trace_enabled": True,
                "awaiting_llm": False,
            }

        if normalized_input == "quiet":
            if trace_enabled:
                print("[TRACE] Quiet command detected. Tracing will be disabled.")
            return {
                "should_exit": False,
                "trace_enabled": False,
                "awaiting_llm": False,
            }

        if normalized_input == "":
            if trace_enabled:
                print("[TRACE] Empty input detected. Routing back to get_user_input.")
            return {
                "should_exit": False,
                "awaiting_llm": False,
            }

        return {
            "messages": [HumanMessage(content=user_input)],
            "should_exit": False,
            "awaiting_llm": True,
        }

    def call_llama(state: AgentState) -> dict:
        """
        Invoke the Llama model using the full chat history stored in state.
        """
        if state.get("trace_enabled", False):
            print("[TRACE] Entering node: call_llama")
            print(f"[TRACE] Conversation length: {len(state['messages'])} message(s)")

        prompt_text = build_prompt_from_messages(
            state["messages"],
            llama_bundle.get("tokenizer"),
        )
        raw_response = llama_bundle["llm"].invoke(prompt_text)
        cleaned_response = clean_model_response(raw_response)

        if state.get("trace_enabled", False):
            print("[TRACE] Llama generation complete.")

        return {
            "messages": [AIMessage(content=cleaned_response)],
            "awaiting_llm": False,
        }

    def print_latest_response(state: AgentState) -> dict:
        """
        Print the latest AI response stored in the message history.
        """
        if state.get("trace_enabled", False):
            print("[TRACE] Entering node: print_latest_response")

        latest_message = state["messages"][-1]

        print("\n" + "-" * 50)
        print("Llama response")
        print("-" * 50)
        print(str(latest_message.content))

        return {}

    def route_after_input(state: AgentState) -> str:
        """
        Decide where to go after get_user_input.

        Returns:
        - END: if the user wants to quit
        - "get_user_input": if the input is empty or was a mode command
        - "call_llama": if a new HumanMessage was added
        """
        if state.get("trace_enabled", False):
            print("[TRACE] Entering router: route_after_input")

        if state.get("should_exit", False):
            if state.get("trace_enabled", False):
                print("[TRACE] Router selected: END")
            return END

        if not state.get("awaiting_llm", False):
            if state.get("trace_enabled", False):
                print("[TRACE] Router selected: get_user_input")
            return "get_user_input"

        if state.get("trace_enabled", False):
            print("[TRACE] Router selected: call_llama")
        return "call_llama"

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llama", call_llama)
    graph_builder.add_node("print_latest_response", print_latest_response)

    graph_builder.add_edge(START, "get_user_input")

    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llama": "call_llama",
            "get_user_input": "get_user_input",
            END: END,
        },
    )

    graph_builder.add_edge("call_llama", "print_latest_response")
    graph_builder.add_edge("print_latest_response", "get_user_input")

    graph = graph_builder.compile()
    return graph


# =============================================================================
# GRAPH VISUALIZATION
# =============================================================================

def save_graph_image(graph, filename: str = "lg_graph.png"):
    """
    Generate a Mermaid diagram of the graph and save it as a PNG image.
    """
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()

        with open(filename, "wb") as file_handle:
            file_handle.write(png_data)

        print(f"Graph image saved to {filename}")
    except Exception as exception_message:
        print(f"Could not save graph image: {exception_message}")
        print("You may need to install additional dependencies: pip install grandalf")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main function that orchestrates the workflow:
    1. Initialize the Llama model only
    2. Create the LangGraph
    3. Save the graph visualization
    4. Run the graph once (it loops internally until the user quits)
    """
    print("=" * 50)
    print("LangGraph Simple Agent with Message History")
    print("=" * 50)
    print()

    llama_model_id = "meta-llama/Llama-3.2-1B-Instruct"

    print("Loading Llama model...")
    llama_bundle = create_llm(llama_model_id)

    print("\nCreating LangGraph...")
    graph = create_graph(llama_bundle)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph)

    initial_state: AgentState = {
        "messages": [
            SystemMessage(
                content="You are a helpful assistant. Keep answers concise and conversational."
            )
        ],
        "should_exit": False,
        "trace_enabled": initial_trace_enabled,
        "awaiting_llm": False,
    }

    if initial_trace_enabled:
        print("Tracing starts in verbose mode. Type 'quiet' to disable it.")
    else:
        print("Tracing starts in quiet mode. Type 'verbose' to enable it.")

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
