"""
Evaluation script for GPT-5-nano dialogue quality metrics.

This script computes automatic evaluation metrics (Entropy, Distinct-N, Self-BLEU, Perplexity)
for GPT-5-nano generated dialogues.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import Counter
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
from nltk.util import ngrams
import numpy as np


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.evaluate.text_process import processed_json_file

# Initialize Qwen2.5-0.5B for perplexity calculation
# Dialogue length: max=5952, P99=4292, P95=3428 tokens
# Qwen2.5-0.5B supports 32K tokens context (32768), covering 100% of dialogues with large margin
# This eliminates the need for sliding window completely
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading Qwen2.5-0.5B model for perplexity calculation...")
print("  Model: Qwen/Qwen2.5-0.5B (32K token context)")
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
perplexity_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
perplexity_model.to(device)
perplexity_model.eval()
print("  Model loaded successfully")


class GPT5NanoEvaluator:
    """Evaluator for GPT-5-nano dialogue quality metrics."""
    
    def __init__(self):
        """Initialize the evaluator with required metrics."""
        self.bleu = evaluate.load("bleu")
        self.results_dir = Path("data/results/evaluation_results/gpt5_nano")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_entropy(self, responses: List[str]) -> float:
        """
        Compute normalized entropy of token distribution.
        
        Args:
            responses: List of response strings
            
        Returns:
            Normalized entropy score (0-1)
        """
        tokens = []
        for resp in responses:
            tokens.extend(resp.split())
        if not tokens:
            return 0.0
        freq = Counter(tokens)
        total = len(tokens)
        entropy = -sum((c / total) * math.log(c / total, 2) for c in freq.values())
        max_entropy = math.log(len(freq), 2) if len(freq) > 1 else 1
        return round(entropy / max_entropy, 4)
    
    def compute_distinct_n(self, responses: List[str], n: int = 2) -> float:
        """
        Compute Distinct-n metric for diversity.
        
        Args:
            responses: List of response strings
            n: n-gram order (default: 2 for bigrams)
            
        Returns:
            Distinct-n score (0-1, higher is more diverse)
        """
        all_ngrams = []
        for response in responses:
            words = response.split()
            if len(words) >= n:
                all_ngrams.extend(list(ngrams(words, n)))
        
        if not all_ngrams:
            return 0.0
        
        ngram_freq = Counter(all_ngrams)
        total_ngrams = len(all_ngrams)
        unique_ngrams = len(ngram_freq)
        
        return round(unique_ngrams / total_ngrams, 4)
    
    def compute_self_bleu(self, generated_responses: List[str]) -> float:
        """
        Compute Self-BLEU score (lower means more diverse).
        
        Args:
            generated_responses: List of generated response strings
            
        Returns:
            Average Self-BLEU score (0-1, lower is more diverse)
        """
        if len(generated_responses) <= 1:
            return 0.0
        
        total_score = 0.0
        comparisons = 0
        
        for i, response in enumerate(generated_responses):
            references = [r for j, r in enumerate(generated_responses) if j != i]
            if references:
                try:
                    # BLEU expects references as list of lists
                    score = self.bleu.compute(
                        predictions=[response],
                        references=[references]
                    )["bleu"]
                    total_score += score
                    comparisons += 1
                except Exception as e:
                    # Skip if computation fails
                    print(f"Warning: Self-BLEU computation error for response {i}: {str(e)}")
                    continue
        
        return round(total_score / max(1, comparisons), 4) if comparisons > 0 else 0.0
    
    def calculate_perplexity(self, text: str, stride: int = 4096, max_length: int = 32768) -> float:
        """
        Calculate perplexity using Qwen2.5-0.5B with 32K token context.
        
        Args:
            text: Input text string
            stride: Stride for sliding window (used only if text exceeds max_length)
            max_length: Maximum sequence length for direct processing
            
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
        
        if total_length <= max_length:
            input_ids = input_ids.unsqueeze(0).to(device)
            labels = input_ids.clone()
            try:
                with torch.no_grad():
                    outputs = perplexity_model(input_ids, labels=labels)
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
                    outputs = perplexity_model(input_ids_slice, labels=target_ids)
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
        actual_length = min(processed_length, total_length)
        ppl = torch.exp(total_nll / actual_length) if actual_length > 0 else 0.0
        return round(ppl.item(), 4)
    
    def dialogue_length(self, conversation: List[str]) -> Dict[str, float]:
        """
        Calculate dialogue length metrics.
        
        Args:
            conversation: List of conversation utterances
            
        Returns:
            Dictionary with avg_length and total_turns
        """
        total_length = 0
        turn = 0
        for utterance in conversation:
            turn += 1
            total_length += len(utterance.split())
        avg_length = round(total_length / turn, 4) if turn > 0 else 0
        return {
            "avg_length": avg_length,
            "total_turns": round(turn / 2, 2)  # Each turn has client + therapist
        }
    
    def evaluate_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single dialogue session.
        
        Args:
            session_data: Session data dictionary from JSON file
            
        Returns:
            Dictionary of evaluation metrics
        """
        dialogue_history = session_data.get("dialogue_history", [])
        
        # Extract therapist responses only
        therapist_responses = [
            turn.get("therapist_utterance", "")
            for turn in dialogue_history
            if turn.get("therapist_utterance", "").strip()
        ]
        
        # Extract all conversation text for perplexity
        conversation_text = []
        for turn in dialogue_history:
            if turn.get("therapist_utterance", "").strip():
                conversation_text.append(turn["therapist_utterance"])
            if turn.get("client_utterance", "").strip():
                conversation_text.append(turn["client_utterance"])
        
        full_text = " ".join(conversation_text)
        
        # Compute metrics
        metrics = {
            "entropy": self.compute_entropy(therapist_responses),
            "distinct_2": self.compute_distinct_n(therapist_responses, n=2),
            "distinct_3": self.compute_distinct_n(therapist_responses, n=3),
            "self_bleu": self.compute_self_bleu(therapist_responses),
            "perplexity": self.calculate_perplexity(full_text),
            **self.dialogue_length(conversation_text)
        }
        
        return metrics
    
    def evaluate_model(self, 
                      session_dir: Path, 
                      condition_name: str,
                      start_index: int = 1,
                      end_index: int = 1001) -> Dict[str, Any]:
        """
        Evaluate all sessions for a given model/condition.
        
        Args:
            session_dir: Directory containing session JSON files
            condition_name: Name of the condition (e.g., "full", "no_story")
            start_index: Start session index (default: 1)
            end_index: End session index (exclusive, default: 1001)
            
        Returns:
            Dictionary of all session results and summary statistics
        """
        all_results = {}
        all_metrics = {
            "entropy": [],
            "distinct_2": [],
            "distinct_3": [],
            "self_bleu": [],
            "perplexity": [],
            "avg_length": [],
            "total_turns": []
        }
        
        print(f"\n{'='*80}")
        print(f"Evaluating: {condition_name}")
        print(f"Session directory: {session_dir}")
        print(f"Range: {start_index} to {end_index-1}")
        print(f"{'='*80}")
        
        for i in range(start_index, end_index):
            session_file = session_dir / f"session_{i}.json"
            
            if not session_file.exists():
                print(f"Warning: Session {i} not found, skipping...")
                continue
            
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                metrics = self.evaluate_session(session_data)
                all_results[f"session_{i}"] = metrics
                
                # Collect metrics for summary
                for key in all_metrics.keys():
                    if key in metrics:
                        all_metrics[key].append(metrics[key])
                
                if (i - start_index + 1) % 100 == 0:
                    print(f"  Processed {i - start_index + 1} sessions...")
                    
            except Exception as e:
                print(f"Error: Processing session {i} failed: {str(e)}")
                continue
        
        # Compute summary statistics
        summary = {}
        for metric_name, values in all_metrics.items():
            if values:
                summary[f"{metric_name}_mean"] = round(np.mean(values), 4)
                summary[f"{metric_name}_std"] = round(np.std(values), 4)
                summary[f"{metric_name}_min"] = round(np.min(values), 4)
                summary[f"{metric_name}_max"] = round(np.max(values), 4)
        
        all_results["summary"] = summary
        all_results["total_sessions"] = len(all_results) - 1  # Exclude summary
        
        print(f"\nCompleted: {all_results['total_sessions']} sessions")
        print(f"   Summary statistics computed")
        
        return all_results
    
    def save_results(self, results: Dict[str, Any], condition_name: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Results dictionary
            condition_name: Name of the condition
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gpt5_nano_{condition_name}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return filepath


def main():
    """Main evaluation function for all GPT-5-nano conditions."""
    
    evaluator = GPT5NanoEvaluator()
    project_root = Path(PROJECT_ROOT)
    
    # Define all conditions to evaluate
    conditions = [
        {
            "name": "full",
            "path": project_root / "data/results/sessions/level1_by2llm/gpt-5-nano",
            "description": "Full condition baseline"
        },
        {
            "name": "ablation_no_story",
            "path": project_root / "data/results/sessions/ablation_no_story/gpt-5-nano",
            "description": "Ablation: No background story"
        },
        {
            "name": "ablation_no_mi_coding_no_story",
            "path": project_root / "data/results/sessions/ablation_no_mi_coding_no_story/gpt-5-nano",
            "description": "Ablation: No MI coding + No background story"
        }
    ]
    
    print("="*80)
    print("GPT-5-nano Dialogue Quality Evaluation")
    print("="*80)
    print("Computing metrics: Entropy, Distinct-N, Self-BLEU, Perplexity")
    print("="*80)
    
    for condition in conditions:
        if not condition["path"].exists():
            print(f"\nWarning: Directory not found: {condition['path']}")
            print(f"   Skipping condition: {condition['name']}")
            continue
        
        results = evaluator.evaluate_model(
            session_dir=condition["path"],
            condition_name=condition["name"],
            start_index=1,
            end_index=1001
        )
        
        evaluator.save_results(results, condition["name"])
    
    print("\n" + "="*80)
    print("All evaluations completed!")
    print("="*80)


if __name__ == "__main__":
    main()

