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
    print("LangGraph Vision Batch Process Started! Type 'quit' to end.")
    print("="*70 + "\n")

    list_start_end_times = []

    while True:
        user_input = input("You (Enter the folder path containing frames): ").strip()
        print("User: ", user_input)
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
            
        if not user_input or not os.path.isdir(user_input):
            print("Please enter a valid directory path.")
            continue
            
        bool_person = False
        start_time = None
        end_time = None

        # 1. SORT the directory so frames are processed in chronological order
        valid_extensions = ('.png', '.jpg', '.jpeg')
        sorted_files = sorted([f for f in os.listdir(user_input) if f.lower().endswith(valid_extensions)])

        for image in sorted_files:
            img_path = os.path.join(user_input, image)
            print(f"[System] Processing: {image}...")
            
            # 2. Extract time from the filename (e.g., 'frame_0005.jpg' -> 5 * 2 = 10s)
            try:
                # Splits by '_' to get '0005.jpg', then splits by '.' to get '0005'
                frame_idx = int(image.split('_')[1].split('.')[0])
                current_time_sec = frame_idx * 2 
            except (IndexError, ValueError):
                print(f"Skipping {image}: Filename does not match expected format.")
                continue

            b64_image = resize_image_for_vlm(img_path)
            
            if b64_image:
                # 3. Corrected message structure (resetting the list per frame)
                message_content = [
                    {"type": "text", "text": "Explicitly return only Yes, if you have a very high confidence that there is a person in the image. Otherwise, return only No. Do NOT return anything else."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            else:
                continue

            # Initialize fresh graph state for this specific frame
            new_message = HumanMessage(content=message_content)
            
            # 4. Use a unique thread ID per frame to prevent context explosion
            frame_config = {"configurable": {"thread_id": f"eval_{image}"}}
            
            # Inject system prompt into the new thread
            app.update_state(frame_config, {"messages": [SystemMessage(content=SYSTEM_PROMPT)]})

            # Stream response
            msg_content = ""
            for event in app.stream({"messages": [new_message]}, config=frame_config):
                for value in event.values():
                    msg_content = value["messages"][-1].content
                    
            # Clean the VLM output to handle edge cases like "Yes." or " Yes "
            clean_response = msg_content.strip().lower()

            print("Is there any person. LLM response: ", clean_response)

            # 5. Start/End Time Tracking Logic
            if "yes" in clean_response:
                if not bool_person:
                    start_time = f"{current_time_sec} seconds"
                    bool_person = True
            elif "no" in clean_response:
                if bool_person:
                    end_time = f"{current_time_sec} seconds"
                    list_start_end_times.append([start_time, end_time])
                    bool_person = False

                    # break # Assuming only one entry/exit event, break the loop to save time

        print("-" * 30)
        print(f"Tracking Complete.")
        print(f"List of enter and exit times. The first and second values presents enter and exit times respectively. {list_start_end_times}")
        print("-" * 30)