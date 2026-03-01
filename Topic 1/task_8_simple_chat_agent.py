"""
Bare-Bones Chat Agent for Llama 3.2-1B-Instruct
(Updated with Context Management and History Toggles)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# CONFIGURATION 
# ============================================================================

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise and friendly."

# TASK 8: Context Management & History Flags

import os

# Reads the notebook environment variable. Defaults to True if not provided.
env_history = os.environ.get("USE_HISTORY", "True").strip().lower()
USE_HISTORY = env_history in ["true"]

MAX_HISTORY_MESSAGES = 6     # Keeps the system prompt + last 3 user/assistant pairs

# ============================================================================
# LOAD MODEL 
# ============================================================================

print("Loading model (this takes 1-2 minutes)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

model.eval()
print(f"✓ Model loaded! Using device: {model.device}")

# Initialize chat history
chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

print("="*70)
print(f"Chat started! (History Enabled: {USE_HISTORY}) Type 'quit' to end.")
print("="*70 + "\n")

while True:
    user_input = input("You: ").strip()

    print('User: ', user_input)
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\nGoodbye!")
        break
    if not user_input:
        continue
    
    # 1. Add new user message
    chat_history.append({"role": "user", "content": user_input})
    
    # ========================================================================
    # 2. CONTEXT MANAGEMENT (Task 8)
    # ========================================================================
    if not USE_HISTORY:
        # If history is disabled, we drop all past interactions.
        # We keep ONLY the System Prompt (index 0) and the Current User Input (the last item)
        chat_history = [chat_history[0], chat_history[-1]]
    else:
        # Sliding Window Truncation: "The simplest option to reduce chat history is to simply sent the last N messages to the LLM".
        # If the list gets too long, we drop the oldest user/assistant pairs but ALWAYS keep the system prompt.
        if len(chat_history) > MAX_HISTORY_MESSAGES + 1:
            # Keep the system prompt at index 0, and slice the most recent N messages
            chat_history = [chat_history[0]] + chat_history[-MAX_HISTORY_MESSAGES:]

    # 3. Tokenize the managed history
    input_ids = tokenizer.apply_chat_template(
        chat_history,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids)

    print("Assistant: ", end="", flush=True)

    # 4. Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 5. Decode response
    new_tokens = outputs[0][input_ids.shape[1]:]
    assistant_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    print(assistant_response)
    
    # 6. Append assistant response so it is available for the next loop
    chat_history.append({"role": "assistant", "content": assistant_response})
    print()