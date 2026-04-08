import json
import os
from pathlib import Path
import sys

# Setup paths
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # …/Trustworthy-LLM-Chatbot-…/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def domain_sanity_check(session_data):
    # --- Extract sections safely ---
    id_domains_raw = session_data.get("identified_domains", [])
    domain_scores_raw = session_data.get("domains", {})

    # 1) normalise domain names to lowercase for comparison
    identified_set = {d.get("domain", "").strip().lower() for d in id_domains_raw if d.get("domain")}
    print("identified_set: ", identified_set)

    # 2) obtain top‑3 domains by domain_score
    scored_items = [
        (name, meta.get("domain_score", 0))
        for name, meta in domain_scores_raw.items()
    ]
    top3 = sorted(scored_items, key=lambda x: x[1], reverse=True)[:3]
    top3_names = [name.lower() for name, _ in top3]

    # 3) Jaccard similarity between identified domains and top‑3 domains
    intersection_len = len(identified_set.intersection(top3_names))
    union_len = len(identified_set.union(top3_names))
    hits = intersection_len  # keep raw hit count for reference
    ratio = intersection_len / union_len if union_len else 0

    return float(f"{ratio:.2f}")

def recalculate_domain_overlap_per_session(eval_file, session_dir):
    eval_path = os.path.join(PROJECT_ROOT, eval_file)
    session_path = os.path.join(PROJECT_ROOT, session_dir) 
    res_path = os.path.join(PROJECT_ROOT, "data/results/evaluation_results/eval1/test.json")

    with open(eval_path, 'r') as f:
        eval_data = json.load(f)  
    
    for i in range(1,1001):
        session_file = os.path.join(session_path, f"session_{i}.json") 
        with open(session_file, 'r') as f:
            session_data = json.load(f)

        # Extract domain overlap
        key = f"session_{i}"
        if eval_data[key].get("Domain Sanity Check") is not None:
            del eval_data[key]["Domain Sanity Check"]
            eval_data[key]["domain_overlap"] = domain_sanity_check(session_data)
    
    # output_path = "data/results/evaluation_results/eval1/evaluation_results_phi4:14b_updated.json"
    # output_path = os.path.join(PROJECT_ROOT, output_path)
    with open(eval_path, 'w') as f:
        json.dump(eval_data, f, indent=4)
        

if __name__ == "__main__":
    name = "qwen3:8b"  # Change this to the desired model name
    eval_file = f"data/results/evaluation_results/eval1/evaluation_results_{name}_updated.json"
    # eval_file = "data/results/evaluation_results/eval1/combined_1_1000.json"
    session_directory = "data/results/sessions/level1_by2llm/" + name

    recalculate_domain_overlap_per_session(eval_file, session_directory)
    print("Domain overlap recalculation completed.")