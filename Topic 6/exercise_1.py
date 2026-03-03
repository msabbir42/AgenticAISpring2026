"""
LangGraph Multi-Turn Vision Agent (LLaVA)
(Features single-input parsing and dynamic file detection)
"""

import os
import base64
from io import BytesIO
from PIL import Image

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_community.chat_models import ChatOllama 

# ============================================================================
# CONFIGURATION
# ============================================================================

# Swapped model to LLaVA based on your configuration
llm = ChatOllama(model="llava") 

SYSTEM_PROMPT = "You are a helpful AI visual assistant. Be concise and friendly."

def resize_image_for_vlm(image_path: str, max_size: tuple = (800, 800)) -> str:
    """Reduces image resolution to prevent the VLM from running slowly."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# ============================================================================
# LANGGRAPH SETUP
# ============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]

def call_model(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ============================================================================
# EXECUTION LOOP
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LangGraph Vision Chat (LLaVA) Started! Type 'quit' to end.")
    print("="*70 + "\n")

    config = {"configurable": {"thread_id": "vision_session_1"}}
    app.update_state(config, {"messages": [SystemMessage(content=SYSTEM_PROMPT)]})

    while True:
        # SINGLE INPUT PROMPT
        user_input = input("You (text or image location: ): ").strip()
        print("User: ", user_input)
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        if not user_input:
            continue
            
        # DYNAMIC PARSING: Separate file paths from text
        img_path = None
        text_parts = []
        
        # Check if the entire input is just an image path (handles paths with spaces)
        clean_input = user_input.strip('"\'')
        if os.path.isfile(clean_input) and clean_input.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = clean_input
            user_text = ""
        else:
            # Fallback: check if the user typed text AND a path (e.g., "Look at /path/img.jpg")
            words = user_input.split()
            for word in words:
                clean_word = word.strip('"\'')
                if os.path.isfile(clean_word) and clean_word.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = clean_word
                else:
                    text_parts.append(word)
            user_text = " ".join(text_parts)
            
        # BUILD THE MESSAGE
        message_content = []
        
        if user_text:
            message_content.append({"type": "text", "text": user_text})
            
        if img_path:
            print(f"[System] Image detected at: {img_path}. Processing...")
            b64_image = resize_image_for_vlm(img_path)
            if b64_image:
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                })

        if not message_content:
            continue

        new_message = HumanMessage(content=message_content)

        # STREAM RESPONSE
        print("Assistant: ", end="", flush=True)
        for event in app.stream({"messages": [new_message]}, config=config):
            for value in event.values():
                print(value["messages"][-1].content)
        print()