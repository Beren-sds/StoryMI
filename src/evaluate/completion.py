"""
Compute completion ratio for every model directory under
data/results/sessions/level1_by2llm and store the results as JSON.

A session counts as *completed* when
    session_metadata["end_reason"] == "Conversation naturally ended"
"""

from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any


# ---------- 1. is a single session completed? ----------
def conversation_completed(session_data: Dict[str, Any]) -> bool:
    """Return True iff the session ended naturally."""
    meta = session_data.get("session_metadata", {})
    return meta.get("completed") == True or len(session_data.get("dialogue_history", [])) == 30


# ---------- 2. main  ----------------------------------------
def compute_completion_ratios(root_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Iterate over all model sub-directories in *root_dir* and compute:

        - total sessions
        - number completed
        - completion_ratio  (completed / total)

    Returns a nested dict keyed by model name.
    """
    results: Dict[str, Dict[str, float]] = {}

    for model_dir in sorted(root_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        session_files = list(model_dir.glob("session_*.json"))

        total = len(session_files)
        completed = 0

        for fp in session_files:
            with fp.open(encoding="utf-8") as f:
                data = json.load(f)
            if conversation_completed(data):
                completed += 1

        ratio = completed / total if total else 0.0
        results[model_name] = {
            "total_sessions": total,
            "completed_sessions": completed,
            "completion_ratio": round(ratio, 4)
        }

    return results


def save_results(payload: Dict[str, Any], out_dir: Path) -> Path:
    """Dump *payload* to a time-stamped JSON file and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"completion_ratio_{ts}.json"
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=4))
    return out_file


if __name__ == "__main__":
    ROOT_SESSIONS = Path("data/results/sessions/level1_by2llm/")
    OUT_DIR = Path("data/results/evaluation_results")

    ratios = compute_completion_ratios(ROOT_SESSIONS)
    outfile = save_results(ratios, OUT_DIR)

    print("Completion ratios written to:", outfile)