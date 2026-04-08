import os, sys
import json
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math


# Import shared components
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # …/Trustworthy-LLM-Chatbot-…/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from src.evaluate.text_process import processed_json_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Qwen2.5-0.5B for perplexity calculation
# Dialogue length: max=5952, P99=4292, P95=3428 tokens
# Qwen2.5-0.5B supports 32K tokens context (32768), covering 100% of dialogues with large margin
# This eliminates the need for sliding window completely
print("Loading Qwen2.5-0.5B model for perplexity calculation...")
print("  Model: Qwen/Qwen2.5-0.5B (32K token context)")
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.to(device)
model.eval()
print("  Model loaded successfully")

def calculate_perplexity(text, stride=4096, max_length=32768):
    """
    Calculate perplexity using Qwen2.5-0.5B with 32K token context.
    
    Args:
        text: Input text string
        stride: Stride for sliding window (used only if text exceeds max_length)
        max_length: Maximum sequence length for direct processing (default: 32768 for Qwen2.5-0.5B)
    
    Returns:
        Perplexity score
    """
    if not text.strip():
        return 0.0
    
    # Tokenize text
    try:
        encodings = tokenizer(text, return_tensors="pt", truncation=False, max_length=None)
        input_ids = encodings["input_ids"][0]
        total_length = input_ids.size(0)
    except Exception as e:
        print(f"Warning: Tokenization error: {str(e)}")
        return 0.0
    
    # If text fits within max_length, process directly without sliding window
    if total_length <= max_length:
        input_ids = input_ids.unsqueeze(0).to(device)
        labels = input_ids.clone()
        try:
            with torch.no_grad():
                outputs = model(input_ids, labels=labels)
                ppl = torch.exp(outputs.loss)
            return round(ppl.item(), 4)
        except Exception as e:
            print(f"Warning: Perplexity computation error: {str(e)}")
            return 0.0
    
    # For extremely long texts (>32K tokens), use sliding window
    nlls = []
    processed_length = 0
    
    for i in range(0, total_length, stride):
        begin = max(i + stride - max_length, 0)
        end = min(i + stride, total_length)
        
        if begin >= end:
            break
            
        input_ids_slice = input_ids[begin:end]
        
        # Ensure slice doesn't exceed max_length
        if input_ids_slice.size(0) > max_length:
            input_ids_slice = input_ids_slice[-max_length:]
            begin = end - max_length
        
        target_ids = input_ids_slice.clone()
        
        # Only compute loss for the new tokens (stride length)
        if i > 0:
            target_ids[:-stride] = -100
        
        input_ids_slice = input_ids_slice.unsqueeze(0).to(device)
        target_ids = target_ids.unsqueeze(0).to(device)
        
        try:
            with torch.no_grad():
                outputs = model(input_ids_slice, labels=target_ids)
                # Calculate loss only for non-ignored tokens
                if i == 0:
                    neg_log_likelihood = outputs.loss * input_ids_slice.size(1)
                else:
                    neg_log_likelihood = outputs.loss * stride
                nlls.append(neg_log_likelihood)
                processed_length += stride if i == 0 else stride
        except Exception as e:
            # Skip if computation fails
            print(f"Warning: Perplexity computation error at position {i}: {str(e)}")
            continue
        
        if end >= total_length:
            break
    
    if not nlls:
        return 0.0
    
    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / processed_length)
    return round(ppl.item(), 4)

def save_evaluation_results(result, output_path, model_name):
    """Save the evaluation results to a JSON file."""
    output_file = output_path / f"perplexity_{model_name}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Results saved to {output_file}")

def main(model_name):
    eval_path = PROJECT_ROOT / f"data/results/sessions/level1_by2llm/{model_name}"
    all_results = {}
    
    for i in range(1, 1001):
        conversation, intent_label, unformatted_dialogue = processed_json_file(i, eval_path)
        conversation = "\n".join(conversation)
        perplexity_score = calculate_perplexity(conversation)
        all_results[i] = perplexity_score
        print(f"Conversation {i} Perplexity: {perplexity_score}")

    # Save all results to a JSON file
    output_path = PROJECT_ROOT / "data/results/evaluation_results"
    output_path.mkdir(parents=True, exist_ok=True)
    save_evaluation_results(all_results, output_path, model_name)
    return all_results

# Example text
if __name__ == "__main__":
    eval_model = "gemma:7b"  # Change this to the model you want to evaluate
    main(eval_model)