import json
import os
import sys
import platform
from datetime import datetime
from datasets import load_dataset
from tqdm.auto import tqdm
import ollama

# ============================================================================
# CONFIGURATION
# ============================================================================
# Use the Ollama model tag format
MODEL_NAME = "llama3.2:1b"

MMLU_SUBJECTS = [
    # "abstract_algebra", "anatomy", 
    "astronomy", "business_ethics",
    # (Uncomment more subjects to run a full test)
]

def check_environment():
    """Check environment and dependencies"""
    print("="*70)
    print("Environment Check")
    print("="*70)
    
    try:
        # Check if Ollama is running by listing models
        ollama.list()
        print("✓ Ollama service is running and accessible")
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("Please ensure the Ollama app is installed and running.")
        sys.exit(1)

    print(f"✓ Platform: {platform.system()} ({platform.machine()})")
    print("\n" + "="*70)
    print("Configuration")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Number of subjects: {len(MMLU_SUBJECTS)}")
    print("="*70 + "\n")

def ensure_model():
    """Ensure the specified model is pulled in Ollama"""
    print(f"\nPulling model {MODEL_NAME} (this will be fast if already downloaded)...")
    try:
        ollama.pull(MODEL_NAME)
        print(f"✓ Model '{MODEL_NAME}' is ready!")
    except Exception as e:
        print(f"❌ Error pulling model: {e}")
        sys.exit(1)

def format_mmlu_prompt(question, choices):
    """Format MMLU question as multiple choice"""
    choice_labels = ["A", "B", "C", "D"]
    prompt = f"{question}\n\n"
    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt

def get_model_prediction(prompt):
    """Get model's prediction for multiple-choice question"""
    response = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt,
        options={
            "temperature": 0.0,
            "num_predict": 1
        }
    )
    
    generated_text = response['response']
    answer = generated_text.strip()[:1].upper()
    
    # Fallback parsing if the model outputs a full sentence instead of a single letter
    if answer not in ["A", "B", "C", "D"]:
        for char in generated_text.upper():
            if char in ["A", "B", "C", "D"]:
                answer = char
                break
        else:
            answer = "A"
    
    return answer

def evaluate_subject(subject):
    """Evaluate model on a specific MMLU subject"""
    print(f"\n{'='*70}")
    print(f"Evaluating subject: {subject}")
    print(f"{'='*70}")
    
    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"❌ Error loading subject {subject}: {e}")
        return None
    
    correct = 0
    total = 0
    
    for example in tqdm(dataset, desc=f"Testing {subject}", leave=True):
        question = example["question"]
        choices = example["choices"]
        correct_answer_idx = example["answer"]
        correct_answer = ["A", "B", "C", "D"][correct_answer_idx]
        
        prompt = format_mmlu_prompt(question, choices)
        predicted_answer = get_model_prediction(prompt)
        
        if predicted_answer == correct_answer:
            correct += 1
        total += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"✓ Result: {correct}/{total} correct = {accuracy:.2f}%")
    
    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "accuracy": accuracy
    }

def main():
    """Main evaluation function"""
    print("\n" + "="*70)
    print(f"{MODEL_NAME} MMLU Evaluation (Ollama)")
    print("="*70 + "\n")

    check_environment()
    ensure_model()
    
    results = []
    total_correct = 0
    total_questions = 0
    
    print(f"\n{'='*70}")
    print(f"Starting evaluation on {len(MMLU_SUBJECTS)} subjects")
    print(f"{'='*70}\n")
    
    start_time = datetime.now()
    
    for i, subject in enumerate(MMLU_SUBJECTS, 1):
        print(f"\nProgress: {i}/{len(MMLU_SUBJECTS)} subjects")
        result = evaluate_subject(subject)
        if result:
            results.append(result)
            total_correct += result["correct"]
            total_questions += result["total"]
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Model: {MODEL_NAME} (via Ollama)")
    print(f"Total Subjects: {len(results)}")
    print(f"Total Questions: {total_questions}")
    print(f"Total Correct: {total_correct}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"Duration: {duration/60:.1f} minutes")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = MODEL_NAME.replace(":", "_")
    output_file = f"{safe_model_name}_mmlu_results_{timestamp}.json"
    
    output_data = {
        "model": MODEL_NAME,
        "timestamp": timestamp,
        "duration_seconds": duration,
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "subject_results": results
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    if len(results) > 0:
        sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
        
        print("\n📊 Top 5 Subjects:")
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {result['subject']}: {result['accuracy']:.2f}%")
        
        print("\n📉 Bottom 5 Subjects:")
        for i, result in enumerate(sorted_results[-5:], 1):
            print(f"  {i}. {result['subject']}: {result['accuracy']:.2f}%")
    
    print("\n✅ Evaluation complete!")
    return output_file

if __name__ == "__main__":
    try:
        output_file = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()