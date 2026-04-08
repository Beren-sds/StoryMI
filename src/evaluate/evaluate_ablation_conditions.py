"""
Evaluation script for GPT-5-nano ablation conditions.

This script evaluates three ablation conditions and compares them with the full condition:
1. Full condition (baseline)
2. Ablation: No story
3. Ablation: No MI coding
4. Ablation: No MI coding + No story

It computes both:
- Automatic metrics (Entropy, Distinct-N, Self-BLEU, Perplexity)
- MI-specific metrics (6 MI-focused metrics)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

# Set environment variable to avoid tokenizer fork deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import evaluators
from src.evaluate.evaluate_gpt5_nano import GPT5NanoEvaluator
from src.evaluate.evaluate_mi_metrics import MIMetricsEvaluator


class AblationEvaluator:
    """Evaluator for ablation conditions comparison."""
    
    def __init__(self):
        """Initialize evaluators."""
        self.auto_evaluator = GPT5NanoEvaluator()
        self.mi_evaluator = MIMetricsEvaluator()
        self.results_dir = Path("data/results/evaluation_results/ablation_comparison")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths to existing results
        self.auto_results_dir = PROJECT_ROOT / "data/results/evaluation_results/gpt5_nano"
        self.mi_results_dir = PROJECT_ROOT / "data/results/evaluation_results/mi_metrics"
        
        # Define all conditions
        self.conditions = [
            {
                "name": "full",
                "path": PROJECT_ROOT / "data/results/sessions/level1_by2llm/gpt-5-nano",
                "description": "Full condition (baseline)"
            },
            {
                "name": "ablation_no_story",
                "path": PROJECT_ROOT / "data/results/sessions/ablation_no_story/gpt-5-nano",
                "description": "Ablation: No background story"
            },
            {
                "name": "ablation_no_mi_coding",
                "path": PROJECT_ROOT / "data/results/sessions/ablation_no_mi_coding/gpt-5-nano",
                "description": "Ablation: No MI coding"
            },
            {
                "name": "ablation_no_mi_coding_no_story",
                "path": PROJECT_ROOT / "data/results/sessions/ablation_no_mi_coding_no_story/gpt-5-nano",
                "description": "Ablation: No MI coding + No background story"
            }
        ]
    
    def find_latest_result_file(self, directory: Path, pattern: str) -> Optional[Path]:
        """
        Find the latest result file matching a pattern.
        
        Args:
            directory: Directory to search in
            pattern: Filename pattern (supports wildcards)
            
        Returns:
            Path to the latest matching file, or None if not found
        """
        if not directory.exists():
            return None
        
        # Try exact pattern first
        exact_match = directory / pattern
        if exact_match.exists():
            return exact_match
        
        # Try glob pattern
        matches = list(directory.glob(pattern))
        if not matches:
            return None
        
        # Return the most recently modified file
        return max(matches, key=lambda p: p.stat().st_mtime)
    
    def load_existing_full_results(self) -> Dict[str, Any]:
        """
        Load existing full condition results.
        
        Returns:
            Dictionary with 'automatic_metrics' and 'mi_metrics' keys
        """
        results = {
            "automatic_metrics": {},
            "mi_metrics": {}
        }
        
        # Load automatic metrics
        print("\nLoading existing full condition automatic metrics...")
        auto_patterns = [
            "gpt5_nano_full_*.json",
            "gpt5_nano_full.json"
        ]
        
        auto_file = None
        for pattern in auto_patterns:
            auto_file = self.find_latest_result_file(self.auto_results_dir, pattern)
            if auto_file:
                break
        
        if auto_file:
            print(f"  Found: {auto_file}")
            try:
                with open(auto_file, 'r', encoding='utf-8') as f:
                    auto_data = json.load(f)
                    results["automatic_metrics"] = auto_data
                    print(f"  [OK] Loaded automatic metrics ({len(auto_data) - 1} sessions + summary)")
            except Exception as e:
                print(f"  [ERROR] Error loading automatic metrics: {e}")
        else:
            print(f"  [WARNING] No existing automatic metrics found in {self.auto_results_dir}")
        
        # Load MI metrics
        print("\nLoading existing full condition MI metrics...")
        mi_patterns = [
            "mi_metrics_gpt-5-nano_*.json",
            "mi_metrics_gpt5_nano_full_*.json",
            "mi_metrics_gpt-5-nano.json"
        ]
        
        mi_file = None
        for pattern in mi_patterns:
            mi_file = self.find_latest_result_file(self.mi_results_dir, pattern)
            if mi_file:
                break
        
        if mi_file:
            print(f"  Found: {mi_file}")
            try:
                with open(mi_file, 'r', encoding='utf-8') as f:
                    mi_data = json.load(f)
                    results["mi_metrics"] = mi_data
                    if "summary" in mi_data:
                        print(f"  [OK] Loaded MI metrics (summary available)")
                    elif "sessions" in mi_data:
                        print(f"  [OK] Loaded MI metrics ({len(mi_data['sessions'])} sessions)")
                    else:
                        print(f"  [OK] Loaded MI metrics")
            except Exception as e:
                print(f"  [ERROR] Error loading MI metrics: {e}")
        else:
            print(f"  [WARNING] No existing MI metrics found in {self.mi_results_dir}")
        
        return results
    
    def evaluate_all_conditions(self, start_index: int = 1, end_index: int = 1001):
        """
        Evaluate all conditions for both automatic and MI-specific metrics.
        Loads existing full condition results instead of recomputing.
        
        Args:
            start_index: Start session index (default: 1)
            end_index: End session index (exclusive, default: 1001)
        """
        print("="*80)
        print("GPT-5-nano Ablation Conditions Evaluation")
        print("="*80)
        print("Evaluating:")
        print("  1. Automatic metrics (Entropy, Distinct-N, Self-BLEU, Perplexity)")
        print("  2. MI-specific metrics (6 MI-focused metrics)")
        print("")
        print("Note: Full condition results will be loaded from existing files.")
        print("="*80)
        
        all_results = {}
        
        # Load existing full condition results
        print("\n" + "="*80)
        print("Loading Full Condition Results")
        print("="*80)
        full_results = self.load_existing_full_results()
        
        if full_results["automatic_metrics"] or full_results["mi_metrics"]:
            all_results["full"] = {
                "description": "Full condition (baseline) - loaded from existing results",
                "automatic_metrics": full_results["automatic_metrics"],
                "mi_metrics": full_results["mi_metrics"]
            }
            print("\n[OK] Full condition results loaded successfully")
        else:
            print("\n[WARNING] Could not load full condition results. Will skip comparison.")
        
        # Evaluate ablation conditions only
        ablation_conditions = [c for c in self.conditions if c["name"] != "full"]
        
        for condition in ablation_conditions:
            condition_name = condition["name"]
            condition_path = condition["path"]
            
            if not condition_path.exists():
                print(f"\nWarning: Directory not found: {condition_path}")
                print(f"   Skipping condition: {condition_name}")
                continue
            
            print(f"\n{'='*80}")
            print(f"Evaluating condition: {condition_name}")
            print(f"Description: {condition['description']}")
            print(f"{'='*80}")
            
            # Evaluate automatic metrics
            print("\n[1/2] Computing automatic metrics...")
            try:
                auto_results = self.auto_evaluator.evaluate_model(
                    session_dir=condition_path,
                    condition_name=condition_name,
                    start_index=start_index,
                    end_index=end_index
                )
            except Exception as e:
                print(f"Error evaluating automatic metrics: {e}")
                auto_results = {}
            
            # Evaluate MI-specific metrics
            print("\n[2/2] Computing MI-specific metrics...")
            try:
                mi_results = self.mi_evaluator.evaluate_model(
                    session_dir=condition_path,
                    model_name=condition_name,
                    start_index=start_index,
                    end_index=end_index
                )
            except Exception as e:
                print(f"Error evaluating MI metrics: {e}")
                mi_results = {}
            
            # Combine results
            all_results[condition_name] = {
                "description": condition["description"],
                "automatic_metrics": auto_results,
                "mi_metrics": mi_results
            }
        
        # Save individual results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"ablation_evaluation_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # Generate comparison
        print("\n" + "="*80)
        print("Generating comparison with full condition...")
        print("="*80)
        comparison = self.generate_comparison(all_results)
        
        comparison_file = self.results_dir / f"ablation_comparison_{timestamp}.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print(f"Comparison saved to: {comparison_file}")
        
        # Print summary
        self.print_comparison_summary(comparison)
        
        return all_results, comparison
    
    def generate_comparison(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comparison between ablation conditions and full condition.
        
        Args:
            all_results: Dictionary of all condition results
            
        Returns:
            Comparison dictionary
        """
        if "full" not in all_results:
            print("Warning: Full condition results not found. Cannot generate comparison.")
            return {}
        
        full_auto = all_results["full"]["automatic_metrics"].get("summary", {})
        full_mi = all_results["full"]["mi_metrics"].get("summary", {})
        
        comparison = {
            "baseline": "full",
            "automatic_metrics_comparison": {},
            "mi_metrics_comparison": {}
        }
        
        # Compare automatic metrics
        for condition_name, condition_data in all_results.items():
            if condition_name == "full":
                continue
            
            auto_summary = condition_data["automatic_metrics"].get("summary", {})
            mi_summary = condition_data["mi_metrics"].get("summary", {})
            
            # Automatic metrics comparison
            auto_comp = {}
            for metric in ["entropy", "distinct_2", "distinct_3", "self_bleu", "perplexity", 
                          "avg_length", "total_turns"]:
                full_key = f"{metric}_mean"
                ablation_key = f"{metric}_mean"
                
                if full_key in full_auto and ablation_key in auto_summary:
                    full_val = full_auto[full_key]
                    ablation_val = auto_summary[ablation_key]
                    diff = ablation_val - full_val
                    diff_pct = (diff / full_val * 100) if full_val != 0 else 0
                    
                    auto_comp[metric] = {
                        "full": full_val,
                        "ablation": ablation_val,
                        "difference": round(diff, 4),
                        "difference_percent": round(diff_pct, 2)
                    }
            
            comparison["automatic_metrics_comparison"][condition_name] = auto_comp
            
            # MI metrics comparison
            mi_comp = {}
            for metric in ["mi_code_entropy", "mi_code_balance_score", "reflection_depth",
                          "complex_reflection_ratio", "question_openness_ratio", 
                          "reflection_to_question_ratio"]:
                full_key = metric
                ablation_key = metric
                
                if full_key in full_mi and ablation_key in mi_summary:
                    full_val = full_mi[full_key].get("mean", 0)
                    ablation_val = mi_summary[ablation_key].get("mean", 0)
                    diff = ablation_val - full_val
                    diff_pct = (diff / full_val * 100) if full_val != 0 else 0
                    
                    mi_comp[metric] = {
                        "full": full_val,
                        "ablation": ablation_val,
                        "difference": round(diff, 4),
                        "difference_percent": round(diff_pct, 2)
                    }
            
            comparison["mi_metrics_comparison"][condition_name] = mi_comp
        
        return comparison
    
    def print_comparison_summary(self, comparison: Dict[str, Any]):
        """Print a summary of the comparison."""
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        if not comparison:
            print("No comparison data available.")
            return
        
        # Automatic metrics
        print("\n--- AUTOMATIC METRICS ---")
        auto_comp = comparison.get("automatic_metrics_comparison", {})
        for condition_name, metrics in auto_comp.items():
            print(f"\n{condition_name.upper()}:")
            for metric_name, values in metrics.items():
                diff = values["difference"]
                diff_pct = values["difference_percent"]
                sign = "+" if diff >= 0 else ""
                print(f"  {metric_name:20s}: Full={values['full']:8.4f}, "
                      f"Ablation={values['ablation']:8.4f}, "
                      f"Diff={sign}{diff:8.4f} ({sign}{diff_pct:6.2f}%)")
        
        # MI metrics
        print("\n--- MI-SPECIFIC METRICS ---")
        mi_comp = comparison.get("mi_metrics_comparison", {})
        for condition_name, metrics in mi_comp.items():
            print(f"\n{condition_name.upper()}:")
            for metric_name, values in metrics.items():
                diff = values["difference"]
                diff_pct = values["difference_percent"]
                sign = "+" if diff >= 0 else ""
                print(f"  {metric_name:30s}: Full={values['full']:8.4f}, "
                      f"Ablation={values['ablation']:8.4f}, "
                      f"Diff={sign}{diff:8.4f} ({sign}{diff_pct:6.2f}%)")
        
        print("\n" + "="*80)


def main():
    """Main evaluation function."""
    evaluator = AblationEvaluator()
    evaluator.evaluate_all_conditions(start_index=1, end_index=1001)


if __name__ == "__main__":
    main()

