"""
LLM-as-a-Judge evaluation script using GPT-5-nano as the judge model.

This script evaluates six models (five open-source models + GPT-5-nano itself)
using GPT-5-nano as the judge.
It evaluates dialogue quality based on evaluation rubrics.
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import OrderedDict

# Add project root to path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.llm import initialize_llm
from src.evaluate.text_process import processed_json_file


class LLMJudgeEvaluator:
    """LLM-as-a-Judge evaluator using GPT-5-nano."""
    
    def __init__(self, judge_model: str = "gpt-5-nano"):
        """
        Initialize the LLM judge evaluator.
        
        Args:
            judge_model: Model name to use as judge (default: gpt-5-nano)
        """
        self.judge_model = judge_model
        self.is_gpt5 = "gpt-5" in judge_model.lower() or "nano" in judge_model.lower()
        
        # Initialize judge LLM
        print(f"Initializing judge model: {judge_model}")
        temp = 1.0 if self.is_gpt5 else 0.3
        self.llm = initialize_llm(
            local_llm=False,
            model_name=judge_model,
            temperature=temp
        )
        
        # Load evaluation rubrics
        rubrics_path = Path(__file__).parent / "evaluation_rubrics.json"
        with open(rubrics_path, "r", encoding='utf-8') as f:
            self.rubrics = json.load(f)
        
        # Create results directory
        self.results_dir = Path("data/results/evaluation_results/llm_judge_gpt5_nano")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Judge model initialized: {judge_model}")
        print(f"   Results directory: {self.results_dir}")
    
    def evaluate_with_llm(self, conversation: List[str]) -> tuple:
        """
        Evaluate conversation using LLM judge.
        
        Args:
            conversation: List of conversation utterances
            
        Returns:
            Tuple of (scores_dict, raw_response)
        """
        # Build rubric prompt (only include rubrics with reference='no')
        # Order: conversation-level metrics first, then therapist-level, then client-level
        rubric_items = [
            (key, value) for key, value in self.rubrics.items()
            if value.get('reference', 'yes') == 'no'
        ]
        # Sort by role: conversation -> therapist -> client
        role_order = {"conversation": 0, "therapist": 1, "client": 2}
        rubric_items.sort(key=lambda x: (role_order.get(x[1].get('role', ''), 99), x[0]))
        
        rubric_prompt = "\n".join([
            f"- {key} ({value['question']})"
            for key, value in rubric_items
        ])
        
        # Build format prompt with consistent ordering
        format_prompt = ", ".join([
            f"{key}: X"
            for key, value in rubric_items
        ])
        
        # Format conversation
        conversation_text = "\n".join(conversation)
        
        prompt = f"""
Evaluate the following therapist-client conversation:
{conversation_text}

Score from 1 to 5 for each criterion:
{rubric_prompt}

Provide scores in this format: {format_prompt}
Without any explanation or additional text.
"""
        
        try:
            response = self.llm.invoke(prompt).content
            
            # Extract scores using regex - handle keys with spaces (e.g., "MI Alignment")
            # Pattern matches: "Key Name: 5" or "Key_Name: 5" or "KeyName: 5"
            matches = re.findall(r'([A-Za-z][A-Za-z_\s]+?):\s*(\d+)\s*', response, re.IGNORECASE)
            scores = {}
            for key, value in matches:
                # Normalize key: remove extra spaces, replace underscores with spaces
                normalized_key = re.sub(r'\s+', ' ', key.strip().replace('_', ' '))
                
                # Match case-insensitive against rubric keys
                matched_rubric_key = None
                for rubric_key in self.rubrics.keys():
                    # Normalize rubric key for comparison
                    normalized_rubric = re.sub(r'\s+', ' ', rubric_key.strip().replace('_', ' '))
                    if normalized_key.lower() == normalized_rubric.lower():
                        matched_rubric_key = rubric_key
                        break
                
                if matched_rubric_key:
                    # Use original rubric key format (preserve capitalization and spacing)
                    scores[matched_rubric_key] = int(value)
            
            return scores, response
            
        except Exception as e:
            print(f"Warning: Error in LLM evaluation: {str(e)}")
            return {}, ""
    
    def evaluate_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single dialogue session.
        
        Args:
            session_data: Session data dictionary from JSON file
            
        Returns:
            Dictionary of evaluation metrics with consistent ordering
        """
        dialogue_history = session_data.get("dialogue_history", [])
        
        # Format conversation
        conversation = []
        for turn in dialogue_history:
            if turn.get("therapist_utterance", "").strip():
                conversation.append(f"Therapist: {turn['therapist_utterance']}")
            if turn.get("client_utterance", "").strip():
                conversation.append(f"Client: {turn['client_utterance']}")
        
        # Evaluate with LLM judge
        llm_scores, llm_response = self.evaluate_with_llm(conversation)
        
        # Order scores by role: conversation -> therapist -> client
        role_order = {"conversation": 0, "therapist": 1, "client": 2}
        ordered_scores = {}
        for key in sorted(
            llm_scores.keys(),
            key=lambda x: (
                role_order.get(self.rubrics.get(x, {}).get('role', ''), 99),
                x
            )
        ):
            ordered_scores[key] = llm_scores[key]
        
        return {
            **ordered_scores,
            "llm_response": llm_response
        }
    
    def evaluate_model(self,
                      session_dir: Path,
                      model_name: str,
                      start_index: int = 1,
                      end_index: int = 1001) -> Dict[str, Any]:
        """
        Evaluate all sessions for a given model.
        
        Args:
            session_dir: Directory containing session JSON files
            model_name: Name of the model being evaluated
            start_index: Start session index (default: 1)
            end_index: End session index (default: 1001)
            
        Returns:
            Dictionary of all session results and summary statistics
        """
        all_results = {}
        all_scores = {}
        
        # Initialize score collectors (only for rubrics with reference='no')
        # Use original rubric key names to preserve formatting (e.g., "MI Alignment")
        for rubric_key, rubric_value in self.rubrics.items():
            if rubric_value.get('reference', 'yes') == 'no':
                all_scores[rubric_key] = []
        
        print(f"\n{'='*80}")
        print(f"Evaluating model: {model_name}")
        print(f"Judge model: {self.judge_model}")
        print(f"Session directory: {session_dir}")
        print(f"Range: {start_index} to {end_index-1}")
        print(f"{'='*80}")
        
        processed = 0
        failed = 0
        
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
                
                # Collect scores for summary
                for key, value in metrics.items():
                    if key in all_scores and isinstance(value, (int, float)):
                        all_scores[key].append(value)
                
                processed += 1
                
                if processed % 50 == 0:
                    print(f"  Processed {processed} sessions...")
                    
            except Exception as e:
                failed += 1
                print(f"Error: Processing session {i} failed: {str(e)}")
                continue
        
        # Compute summary statistics
        # Sort metrics by role order for consistent output
        role_order = {"conversation": 0, "therapist": 1, "client": 2}
        sorted_metrics = sorted(
            all_scores.items(),
            key=lambda x: (
                role_order.get(self.rubrics.get(x[0], {}).get('role', ''), 99),
                x[0]
            )
        )
        
        summary = {}
        for metric_name, values in sorted_metrics:
            if values:
                summary[f"{metric_name}_mean"] = round(sum(values) / len(values), 4)
                summary[f"{metric_name}_std"] = round(
                    (sum((x - summary[f"{metric_name}_mean"])**2 for x in values) / len(values))**0.5,
                    4
                )
                summary[f"{metric_name}_min"] = min(values)
                summary[f"{metric_name}_max"] = max(values)
        
        all_results["summary"] = summary
        all_results["total_sessions"] = processed
        all_results["failed_sessions"] = failed
        
        print(f"\nCompleted: {processed} sessions, {failed} failed")
        print(f"   Summary statistics computed")
        
        return all_results
    
    def save_results(self, results: Dict[str, Any], model_name: str, condition: str = None):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Results dictionary
            model_name: Name of the evaluated model
            condition: Optional condition name (e.g., "ablation_no_story") for ablation studies
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_clean = model_name.replace(':', '_').replace('-', '_')
        judge_name_clean = self.judge_model.replace('-', '_').replace(':', '_')
        
        if condition:
            filename = f"llm_judge_{judge_name_clean}_{model_name_clean}_{condition}_{timestamp}.json"
        else:
            filename = f"llm_judge_{judge_name_clean}_{model_name_clean}_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return filepath


def evaluate_all_models():
    """Evaluate all models (five open-source + GPT-5-nano full condition)."""
    
    evaluator = LLMJudgeEvaluator(judge_model="gpt-5-nano")
    project_root = Path(PROJECT_ROOT)
    
    # Models to evaluate: five open-source models + GPT-5-nano itself
    models = [
        "llama3.1:8b",
        "phi4:14b",
        "openchat:7b",
        "gemma:7b",
        "qwen2.5:7b",
        "gpt-5-nano"  # Add GPT-5-nano to evaluate itself
    ]
    
    print("="*80)
    print("LLM-as-a-Judge Evaluation - All Models")
    print("="*80)
    print(f"Judge Model: gpt-5-nano")
    print(f"Models to evaluate: {', '.join(models)}")
    print("="*80)
    
    all_model_results = {}
    
    for model_name in models:
        # All models are in level1_by2llm directory
        session_dir = project_root / "data/results/sessions/level1_by2llm" / model_name
        
        if not session_dir.exists():
            print(f"\nWarning: Directory not found: {session_dir}")
            print(f"   Skipping model: {model_name}")
            continue
        
        results = evaluator.evaluate_model(
            session_dir=session_dir,
            model_name=model_name,
            start_index=1,
            end_index=1001
        )
        
        evaluator.save_results(results, model_name)
        
        # Store summary for combined report
        all_model_results[model_name] = {
            "summary": results.get("summary", {}),
            "total_sessions": results.get("total_sessions", 0),
            "failed_sessions": results.get("failed_sessions", 0)
        }
    
    # Save combined summary
    summary_file = evaluator.results_dir / f"llm_judge_gpt5_nano_all_models_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_model_results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*80)
    print("All LLM judge evaluations completed!")
    print(f"Summary saved to: {summary_file}")
    print("="*80)


def evaluate_ablation_conditions():
    """Evaluate GPT-5-nano ablation conditions."""
    
    evaluator = LLMJudgeEvaluator(judge_model="gpt-5-nano")
    project_root = Path(PROJECT_ROOT)
    
    # Define ablation conditions
    ablation_conditions = [
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
        # }
    ]
    
    print("="*80)
    print("LLM-as-a-Judge Evaluation - GPT-5-nano Ablation Conditions")
    print("="*80)
    print(f"Judge Model: gpt-5-nano")
    print(f"Evaluating GPT-5-nano in ablation conditions")
    print("="*80)
    
    all_ablation_results = {}
    
    for condition in ablation_conditions:
        if not condition["path"].exists():
            print(f"\nWarning: Directory not found: {condition['path']}")
            print(f"   Skipping condition: {condition['name']}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Evaluating condition: {condition['name']}")
        print(f"Description: {condition['description']}")
        print(f"{'='*80}")
        
        results = evaluator.evaluate_model(
            session_dir=condition["path"],
            model_name="gpt-5-nano",
            start_index=1,
            end_index=1001
        )
        
        evaluator.save_results(results, "gpt-5-nano", condition=condition["name"])
        
        # Store summary
        all_ablation_results[condition["name"]] = {
            "summary": results.get("summary", {}),
            "total_sessions": results.get("total_sessions", 0),
            "failed_sessions": results.get("failed_sessions", 0)
        }
    
    # Save combined summary for ablation conditions
    if all_ablation_results:
        summary_file = evaluator.results_dir / f"llm_judge_gpt5_nano_ablation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_ablation_results, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*80)
        print("All ablation condition evaluations completed!")
        print(f"Ablation summary saved to: {summary_file}")
        print("="*80)


def main():
    """Main evaluation function - evaluates all models and ablation conditions."""
    
    # First evaluate all models (including GPT-5-nano full condition)
    evaluate_all_models()
    
    # Then evaluate GPT-5-nano ablation conditions
    print("\n\n")
    evaluate_ablation_conditions()
    
    print("\n" + "="*80)
    print("All evaluations completed!")
    print("="*80)


if __name__ == "__main__":
    main()

