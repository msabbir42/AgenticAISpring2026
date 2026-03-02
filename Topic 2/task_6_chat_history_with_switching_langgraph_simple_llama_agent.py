# langgraph_simple_agent.py
# Program demonstrates use of LangGraph for a simple multi-speaker agent.
# It writes to stdout and asks the user to enter a line of text through stdin.
# It can switch between a Llama model and a Qwen model while preserving chat history
# with the LangGraph/LangChain Message API.

import os
from typing import TypedDict

import torch
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import END, START, MessagesState, StateGraph
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def decide_whether_to_print(env_var_name: str, default_value: bool = False) -> bool:
    """Read an environment variable and convert it to a verbose/quiet boolean."""
    env_value = os.getenv(env_var_name)

    if env_value is None:
        return default_value

    normalized_value = env_value.strip().lower()

    if normalized_value in {"verbose", "true", "1", "yes", "on"}:
        return True

    if normalized_value in {"quiet", "false", "0", "no", "off"}:
        return False

    return default_value


bool_verbose = decide_whether_to_print("bool_verbose", default_value=True)


def get_device() -> str:
    """Detect and return the best available compute device."""
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"

    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"

    print("Using CPU for inference")
    return "cpu"


class AgentState(MessagesState):
    """State object flowing through the LangGraph nodes."""

    should_exit: bool
    trace_enabled: bool
    awaiting_llm: bool
    pending_target: str


SPEAKER_NAMES = ("Human", "Llama", "Qwen")


def create_llm_bundle(model_id: str) -> dict:
    device = get_device()
    print(f"Loading model: {model_id}")
    print("This may take a moment on first run as the model is downloaded...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Apply the eager attention fix and correct torch_dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32 if device == "mps" else torch.float16,
        attn_implementation="eager" if device == "mps" else None,
        device_map=device if device == "cuda" else None,
    )

    if device == "mps":
        model = model.to(device)

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=text_pipeline)

    print("Model loaded successfully!")
    return {
        "model_id": model_id,
        "tokenizer": tokenizer,
        "llm": llm,
    }


def parse_speaker_and_text(message_content: str) -> tuple[str, str]:
    """Split a stored utterance like 'Llama: hello' into speaker and text."""
    content_text = str(message_content).strip()

    for speaker_name in SPEAKER_NAMES:
        prefix_text = f"{speaker_name}:"
        if content_text.startswith(prefix_text):
            return speaker_name, content_text[len(prefix_text) :].strip()

    return "Human", content_text


def build_system_prompt(target_model_name: str) -> str:
    """Build the model-specific system prompt describing all participants."""
    other_model_name = "Qwen" if target_model_name == "Llama" else "Llama"
    return (
        f"You are {target_model_name}. There are three participants in this conversation: "
        f"Human, Llama, and Qwen. The Human is the real user. {other_model_name} is another AI participant. "
        "All messages are prefixed with the speaker's name. "
        f"Messages from Human and {other_model_name} will appear as user messages. "
        f"Your own previous messages will appear as assistant messages and are prefixed with '{target_model_name}:'. "
        f"Reply as {target_model_name}, keep the conversation coherent, and do not invent extra speakers."
    )


def convert_message_to_role_and_content(message: BaseMessage) -> tuple[str, str]:
    """Convert a LangChain message object into a role/content pair for chat templating."""
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


def build_messages_for_target(history_messages: list[BaseMessage], target_model_name: str) -> list[BaseMessage]:
    """Create a target-specific message list with model-aware role remapping."""
    prompt_messages: list[BaseMessage] = [
        SystemMessage(content=build_system_prompt(target_model_name))
    ]

    for history_message in history_messages:
        if isinstance(history_message, SystemMessage):
            continue

        speaker_name, utterance_text = parse_speaker_and_text(history_message.content)
        named_text = f"{speaker_name}: {utterance_text}"

        if speaker_name == target_model_name:
            prompt_messages.append(AIMessage(content=named_text))
        else:
            prompt_messages.append(HumanMessage(content=named_text))

    return prompt_messages


def build_prompt_from_messages(messages: list[BaseMessage], tokenizer) -> str:
    """Convert structured messages into a model prompt string."""
    chat_messages: list[dict[str, str]] = []

    for message in messages:
        role_name, content_text = convert_message_to_role_and_content(message)

        if role_name == "function":
            role_name = "tool"

        chat_messages.append({"role": role_name, "content": content_text})

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    fallback_lines: list[str] = []

    for chat_message in chat_messages:
        role_name = chat_message["role"]
        content_text = chat_message["content"]
        fallback_lines.append(f"{role_name.title()}: {content_text}")

    fallback_lines.append("Assistant:")
    return "\n".join(fallback_lines)


def clean_model_response(response_text: str, target_model_name: str) -> str:
    """Trim model output so only the current speaker's answer remains."""
    cleaned_text = str(response_text).strip()

    own_prefix = f"{target_model_name}:"
    if cleaned_text.startswith(own_prefix):
        cleaned_text = cleaned_text[len(own_prefix) :].strip()

    for stop_prefix in ("\nHuman:", "\nLlama:", "\nQwen:", "\nUser:", "\nAssistant:"):
        if stop_prefix in cleaned_text:
            cleaned_text = cleaned_text.split(stop_prefix, 1)[0].strip()

    return cleaned_text


def create_graph(llama_bundle: dict, qwen_bundle: dict):
    """Create the LangGraph with history-aware switching between Llama and Qwen."""

    def trace_print(state: AgentState, message_text: str) -> None:
        if state.get("trace_enabled", False):
            print(f"[TRACE] {message_text}")

    def get_user_input(state: AgentState) -> dict:
        print("Enter your text:")

        user_input = input()
        print(f"User: {user_input}")

        stripped_input = user_input.strip()
        normalized_input = stripped_input.lower()

        if normalized_input in {"quit", "exit", "q"}:
            trace_print(state, "User requested exit")
            return {
                "should_exit": True,
                "awaiting_llm": False,
                "pending_target": "",
            }

        if normalized_input == "verbose":
            print("Tracing enabled.")
            return {
                "trace_enabled": True,
                "awaiting_llm": False,
                "pending_target": "",
            }

        if normalized_input == "quiet":
            return {
                "trace_enabled": False,
                "awaiting_llm": False,
                "pending_target": "",
            }

        if stripped_input == "":
            trace_print(state, "Empty input detected; looping back to get_user_input")
            return {
                "awaiting_llm": False,
                "pending_target": "",
            }

        pending_target = "qwen" if normalized_input.startswith("hey qwen") else "llama"
        trace_print(state, f"Queued input for {pending_target}")

        return {
            "messages": [HumanMessage(content=f"Human: {user_input}")],
            "should_exit": False,
            "awaiting_llm": True,
            "pending_target": pending_target,
        }

    def call_llama(state: AgentState) -> dict:
        trace_print(state, "Entering node: call_llama")

        prompt_messages = build_messages_for_target(state["messages"], "Llama")
        prompt_text = build_prompt_from_messages(
            prompt_messages,
            llama_bundle["tokenizer"],
        )

        response_text = llama_bundle["llm"].invoke(prompt_text)
        cleaned_response = clean_model_response(response_text, "Llama")

        trace_print(state, f"Llama response ready: {cleaned_response}")
        return {
            "messages": [AIMessage(content=f"Llama: {cleaned_response}")],
            "awaiting_llm": False,
            "pending_target": "",
        }

    def call_qwen(state: AgentState) -> dict:
        trace_print(state, "Entering node: call_qwen")

        prompt_messages = build_messages_for_target(state["messages"], "Qwen")
        prompt_text = build_prompt_from_messages(
            prompt_messages,
            qwen_bundle["tokenizer"],
        )

        response_text = qwen_bundle["llm"].invoke(prompt_text)
        cleaned_response = clean_model_response(response_text, "Qwen")

        trace_print(state, f"Qwen response ready: {cleaned_response}")
        return {
            "messages": [AIMessage(content=f"Qwen: {cleaned_response}")],
            "awaiting_llm": False,
            "pending_target": "",
        }

    def print_latest_response(state: AgentState) -> dict:
        latest_message = state["messages"][-1]
        speaker_name, utterance_text = parse_speaker_and_text(latest_message.content)

        if state.get("trace_enabled", False):
            print("\n" + "-" * 50)
            print(f"Latest response from {speaker_name}")
            print("-" * 50)

        print(f"{speaker_name}: {utterance_text}")
        return {}

    def route_after_input(state: AgentState) -> str:
        if state.get("should_exit", False):
            return END

        if not state.get("awaiting_llm", False):
            return "get_user_input"

        if state.get("pending_target") == "qwen":
            return "call_qwen"

        return "call_llama"

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llama", call_llama)
    graph_builder.add_node("call_qwen", call_qwen)
    graph_builder.add_node("print_latest_response", print_latest_response)

    graph_builder.add_edge(START, "get_user_input")

    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "get_user_input": "get_user_input",
            "call_llama": "call_llama",
            "call_qwen": "call_qwen",
            END: END,
        },
    )

    graph_builder.add_edge("call_llama", "print_latest_response")
    graph_builder.add_edge("call_qwen", "print_latest_response")
    graph_builder.add_edge("print_latest_response", "get_user_input")

    return graph_builder.compile()


def save_graph_image(graph, filename: str = "lg_graph.png") -> None:
    """Generate a Mermaid diagram of the graph and save it as a PNG image."""
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()

        with open(filename, "wb") as file_handle:
            file_handle.write(png_data)

        print(f"Graph image saved to {filename}")
    except Exception as exc:
        print(f"Could not save graph image: {exc}")
        print("You may need to install additional dependencies: pip install grandalf")


def main() -> None:
    """Load both models, build the graph, and run the interactive loop."""
    print("=" * 50)
    print("LangGraph Multi-Speaker Agent with Llama and Qwen")
    print("=" * 50)
    print()

    llama_bundle = create_llm_bundle("meta-llama/Llama-3.2-1B-Instruct")
    qwen_bundle = create_llm_bundle("Qwen/Qwen2.5-0.5B")

    print("\nCreating LangGraph...")
    graph = create_graph(llama_bundle, qwen_bundle)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph)

    initial_state: AgentState = {
        "messages": [],
        "should_exit": False,
        "trace_enabled": bool_verbose,
        "awaiting_llm": False,
        "pending_target": "",
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
