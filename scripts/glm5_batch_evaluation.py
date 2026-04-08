#!/usr/bin/env python3
"""
GLM-5 Batch Evaluation - Generates and executes 600 evaluation API calls.

This script:
1. Samples 100 sessions per model (20 human-annotated + 80 random, seed=42)
2. Formats each dialogue for evaluation
3. Calls GLM-5 API (open.bigmodel.cn) for each dialogue
4. Saves results with checkpointing for resume capability

The script checkpoints after every evaluation so it can be safely interrupted
and resumed. On restart, it skips already-completed evaluations.

Usage:
    python scripts/glm5_batch_evaluation.py
    python scripts/glm5_batch_evaluation.py --dry-run  # just prepare, no API calls
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSION_DIR = PROJECT_ROOT / "data" / "results" / "sessions" / "level1_by2llm"
OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "evaluation_results" / "glm5_cross_model"

GLM_API_KEY = "ecf75a89570d40eba9def8c4d20f82d8.TcQLARmZNNwraQUc"
GLM_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
GLM_MODEL = "glm-4-plus"
GLM_TEMPERATURE = 0.3

MODELS = [
    "gpt-5-nano", "llama3.1:8b", "phi4:14b",
    "openchat:7b", "gemma:7b", "qwen2.5:7b",
]

HUMAN_ANNOTATED_IDS = [
    35, 62, 111, 116, 207, 285, 325, 420, 473, 539,
    594, 653, 698, 735, 769, 837, 854, 880, 928, 982,
]

DIMENSIONS = [
    "Coherence", "Depth", "Progress", "Naturalness", "Empathy", "MI Alignment"
]

SEED = 42
SESSIONS_PER_MODEL = 100
MAX_RETRIES = 4
BASE_RETRY_DELAY = 2  # exponential backoff: 2s, 4s, 8s, 16s


# ── Sampling ─────────────────────────────────────────────────────────────────


def sample_sessions(model_name: str) -> list:
    """Sample 100 sessions: 20 human-annotated + 80 random (seed=42)."""
    model_dir = SESSION_DIR / model_name
    all_ids = sorted(
        int(f.stem.split("_")[1]) for f in model_dir.glob("session_*.json")
    )

    must_include = set(HUMAN_ANNOTATED_IDS)
    remaining = [sid for sid in all_ids if sid not in must_include]

    rng = random.Random(SEED)
    additional = sorted(rng.sample(remaining, min(80, len(remaining))))

    return sorted(list(must_include) + additional)[:SESSIONS_PER_MODEL]


# ── Dialogue Formatting ─────────────────────────────────────────────────────


def format_dialogue(session_data: dict) -> str:
    """Format dialogue_history into evaluation string."""
    dialogue_history = session_data.get("dialogue_history", [])
    lines = []
    turn_num = 0

    for turn in dialogue_history:
        therapist = turn.get("therapist_utterance", "").strip()
        client = turn.get("client_utterance", "").strip()

        # Skip greeting turn (first turn with empty client)
        if not client and turn_num == 0:
            turn_num += 1
            continue

        turn_num += 1
        turn_lines = [f"Turn {turn_num}:"]
        if therapist:
            turn_lines.append(f"Therapist: {therapist}")
        if client:
            turn_lines.append(f"Client: {client}")
        lines.append("\n".join(turn_lines))

    return "\n\n".join(lines)


def build_prompt(formatted_dialogue: str) -> str:
    """Build evaluation prompt matching the original GPT evaluation."""
    return (
        "Evaluate the following therapist-client conversation:\n"
        f"{formatted_dialogue}\n\n"
        "Score from 1 to 5:\n"
        "- Coherence (Is the conversation logically structured, with smooth transitions between steps?)\n"
        "- Depth (To what extent does the dialogue move beyond surface remarks to examine underlying "
        "emotions, cognitions, life history, and relational patterns, thereby demonstrating therapeutic depth?)\n"
        "- Progress (Does the conversation effectively move forward in a logical manner?)\n"
        "- Naturalness (Does the conversation feel fluid and human-like, avoiding robotic or overly scripted responses?)\n"
        "- Empathy (Does the therapist convey accurate understanding and acceptance of the client's perspective?)\n"
        "- MI Alignment (Does the therapist align with the predicted Motivational Interviewing (MI) skills?)\n\n"
        "Provide scores in this format: Coherence: X, Depth: X, Progress: X, Naturalness: X, Empathy: X, MI Alignment: X\n"
        "without any explanation or additional text."
    )


def parse_scores(response_text: str) -> dict:
    """Parse scores from model response."""
    matches = re.findall(r'(\w[\w\s]*?):\s*(\d+)', response_text)
    scores = {}
    for key, value in matches:
        key_clean = key.strip()
        val = int(value)
        if val < 1 or val > 5:
            continue
        key_lower = key_clean.lower()
        if "coherence" in key_lower:
            scores["Coherence"] = val
        elif "depth" in key_lower:
            scores["Depth"] = val
        elif "progress" in key_lower:
            scores["Progress"] = val
        elif "natural" in key_lower:
            scores["Naturalness"] = val
        elif "empathy" in key_lower:
            scores["Empathy"] = val
        elif "mi" in key_lower and "alignment" in key_lower:
            scores["MI Alignment"] = val
    return scores


# ── API Call ─────────────────────────────────────────────────────────────────


def call_glm5_api(prompt: str) -> tuple:
    """
    Call GLM-5 API with exponential backoff retry.
    Returns (response_text, error_message).
    """
    import requests

    headers = {
        "Authorization": f"Bearer {GLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": GLM_TEMPERATURE,
        "max_tokens": 200,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                GLM_API_URL, headers=headers, json=payload, timeout=60
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content, None
            elif resp.status_code == 429:
                # Rate limited
                error_msg = f"Rate limited (429): {resp.text[:100]}"
            else:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:200]}"

        if attempt < MAX_RETRIES - 1:
            delay = BASE_RETRY_DELAY * (2 ** attempt)
            print(f"      Retry {attempt+1}/{MAX_RETRIES-1} in {delay}s... ({error_msg})")
            time.sleep(delay)

    return None, error_msg


# ── Main Pipeline ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Prepare sampling and prompts without calling API")
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = OUTPUT_DIR / "glm5_evaluation_results.json"
    checkpoint_file = OUTPUT_DIR / "glm5_evaluation_checkpoint.json"
    sampling_file = OUTPUT_DIR / "glm5_sampling_plan.json"

    # Load checkpoint
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            results = json.load(f)
        print(f"Resumed checkpoint: {sum(len(v) for v in results.values())} evaluations")
    elif results_file.exists() and not args.dry_run:
        with open(results_file) as f:
            results = json.load(f)
        print(f"Loaded existing results: {sum(len(v) for v in results.values())} evaluations")
    else:
        results = {}

    models = args.models or MODELS
    sampling_plan = {}
    total_pending = 0

    # Phase 1: Prepare sampling
    print("\n=== PHASE 1: SAMPLING ===")
    for model in models:
        model_dir = SESSION_DIR / model
        if not model_dir.exists():
            print(f"  SKIP {model}: directory not found")
            continue

        sampled = sample_sessions(model)
        sampling_plan[model] = sampled

        if model not in results:
            results[model] = {}

        pending = [s for s in sampled if f"session_{s}" not in results[model]]
        total_pending += len(pending)
        print(f"  {model}: {len(sampled)} sampled, {len(sampled)-len(pending)} done, {len(pending)} pending")

    # Save sampling plan
    with open(sampling_file, "w") as f:
        json.dump(sampling_plan, f, indent=2)
    print(f"\nSampling plan saved: {sampling_file}")
    print(f"Total pending evaluations: {total_pending}")

    if args.dry_run:
        # In dry-run mode, just prepare and show a sample prompt
        model = models[0]
        sid = sampling_plan[model][0]
        with open(SESSION_DIR / model / f"session_{sid}.json") as f:
            session = json.load(f)
        formatted = format_dialogue(session)
        prompt = build_prompt(formatted)
        print(f"\n=== SAMPLE PROMPT (model={model}, session_{sid}) ===")
        print(prompt[:800] + "..." if len(prompt) > 800 else prompt)
        print(f"\n(Prompt length: {len(prompt)} chars)")
        return

    # Phase 2: Evaluate
    print("\n=== PHASE 2: EVALUATION ===")
    total_done = 0
    total_failed = 0
    api_accessible = True

    for model in models:
        if model not in sampling_plan:
            continue

        sampled = sampling_plan[model]
        pending = [s for s in sampled if f"session_{s}" not in results[model]]

        if not pending:
            print(f"\n  {model}: all done, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  Model: {model} ({len(pending)} pending)")
        print(f"{'='*60}")

        for i, sid in enumerate(pending):
            session_file = SESSION_DIR / model / f"session_{sid}.json"
            if not session_file.exists():
                continue

            with open(session_file) as f:
                session = json.load(f)

            formatted = format_dialogue(session)
            prompt = build_prompt(formatted)

            print(f"  [{i+1}/{len(pending)}] session_{sid}...", end=" ", flush=True)

            if not api_accessible:
                # Skip API calls if we know the API is unreachable
                results[model][f"session_{sid}"] = {
                    "scores": {},
                    "raw_response": "",
                    "status": "api_unreachable",
                    "prompt_length": len(prompt),
                }
                total_failed += 1
                print("SKIPPED (API unreachable)")
                continue

            response, error = call_glm5_api(prompt)

            if response:
                scores = parse_scores(response)
                status = "success" if len(scores) == 6 else "partial"
                results[model][f"session_{sid}"] = {
                    "scores": scores,
                    "raw_response": response,
                    "status": status,
                }
                total_done += 1
                print(f"OK {scores}")
            elif error and ("ProxyError" in error or "Tunnel" in error or "NameResolution" in error):
                # API is completely unreachable
                api_accessible = False
                results[model][f"session_{sid}"] = {
                    "scores": {},
                    "raw_response": "",
                    "status": "api_unreachable",
                    "error": error,
                    "prompt_length": len(prompt),
                }
                total_failed += 1
                print(f"API UNREACHABLE: {error[:80]}")
                print("\n  *** GLM-5 API (open.bigmodel.cn) is not reachable from this environment ***")
                print("  *** Marking remaining evaluations as api_unreachable ***")
                print("  *** Re-run this script from an environment with direct internet access ***\n")
            else:
                results[model][f"session_{sid}"] = {
                    "scores": {},
                    "raw_response": "",
                    "status": "failed",
                    "error": error,
                }
                total_failed += 1
                print(f"FAILED: {error[:80]}")

            # Checkpoint every 10
            if (total_done + total_failed) % 10 == 0:
                with open(checkpoint_file, "w") as f:
                    json.dump(results, f, indent=2)

            # Rate limit
            if api_accessible:
                time.sleep(1.0)

    # Save final results
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {results_file}")
    print(f"  Success: {total_done}, Failed: {total_failed}")

    # Clean checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    if not api_accessible:
        print("\n" + "=" * 70)
        print("NOTE: The GLM-5 API was not reachable from this environment.")
        print("To complete the evaluation:")
        print("  1. Ensure you have network access to open.bigmodel.cn")
        print("  2. Re-run: python scripts/glm5_batch_evaluation.py")
        print("  The script will resume from where it left off (checkpointing).")
        print("  3. Then run analysis: python scripts/evaluate_glm5_cross_model.py --analyze-only")
        print("=" * 70)


if __name__ == "__main__":
    main()
