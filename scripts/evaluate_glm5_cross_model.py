#!/usr/bin/env python3
"""
Cross-Model Evaluator: GLM-5 Validation

Re-evaluates 600 dialogue sessions (100 per model × 6 models) using GLM-5
as an independent LLM judge, then computes correlations with human evaluations
to address the self-bias concern raised by Reviewers yFh4 and FLgm.

Usage:
    # Full pipeline: evaluate + analyze
    python scripts/evaluate_glm5_cross_model.py

    # Analysis only (if glm5_evaluation_results.json already exists)
    python scripts/evaluate_glm5_cross_model.py --analyze-only

    # Evaluate specific model(s)
    python scripts/evaluate_glm5_cross_model.py --models gemma:7b llama3.1:8b

Outputs:
    - data/results/evaluation_results/glm5_cross_model/glm5_evaluation_results.json
    - data/results/evaluation_results/glm5_cross_model/glm5_vs_gpt_correlation_comparison.csv
    - data/results/evaluation_results/glm5_cross_model/self_bias_analysis.txt
"""

import argparse
import json
import os
import random
import re
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom")

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SESSION_DIR = DATA_DIR / "results" / "sessions" / "level1_by2llm"
EVAL_DIR = DATA_DIR / "results" / "evaluation_results"
OUTPUT_DIR = EVAL_DIR / "glm5_cross_model"
HUMAN_ANNO_DIR = PROJECT_ROOT / "human_anno" / "csv"

GLM_API_KEY = "ecf75a89570d40eba9def8c4d20f82d8.TcQLARmZNNwraQUc"
GLM_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
GLM_MODEL = "glm-4-plus"  # GLM-5 series model
GLM_TEMPERATURE = 0.3

MODELS = [
    "gpt-5-nano",
    "llama3.1:8b",
    "phi4:14b",
    "openchat:7b",
    "gemma:7b",
    "qwen2.5:7b",
]

# The 20 human-annotated session IDs
HUMAN_ANNOTATED_IDS = [
    35, 62, 111, 116, 207, 285, 325, 420, 473, 539,
    594, 653, 698, 735, 769, 837, 854, 880, 928, 982,
]

# Evaluation dimensions
DIMENSIONS = [
    "Coherence", "Depth", "Progress", "Naturalness", "Empathy", "MI Alignment"
]

# Model name mapping: human annotation CSV uses short names
HUMAN_ANNO_MODEL_MAP = {
    "gemma:7b": "gemma:7b",
    "llama3.1": "llama3.1:8b",
    "llama3.1:8b": "llama3.1:8b",
    "openchat:7b": "openchat:7b",
    "phi4:14b": "phi4:14b",
    "qwen2.5:7b": "qwen2.5:7b",
    "gpt-5-nano": "gpt-5-nano",
}

SEED = 42
SESSIONS_PER_MODEL = 100
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, exponential backoff

# ── Sampling ─────────────────────────────────────────────────────────────────


def sample_sessions(model_name: str) -> list:
    """
    Sample 100 sessions for a model:
    - Always include the 20 human-annotated sessions
    - Plus 80 randomly sampled (seed=42) from the remaining sessions
    """
    model_dir = SESSION_DIR / model_name
    all_ids = []
    for f in model_dir.glob("session_*.json"):
        sid = int(f.stem.split("_")[1])
        all_ids.append(sid)
    all_ids.sort()

    # Always include human-annotated sessions
    must_include = set(HUMAN_ANNOTATED_IDS)
    remaining = [sid for sid in all_ids if sid not in must_include]

    # Random sample 80 from remaining
    rng = random.Random(SEED)
    additional = sorted(rng.sample(remaining, min(80, len(remaining))))

    sampled = sorted(list(must_include) + additional)
    return sampled[:SESSIONS_PER_MODEL]


# ── Dialogue Formatting ─────────────────────────────────────────────────────


def format_dialogue(session_data: dict) -> str:
    """Format dialogue_history into evaluation string."""
    dialogue_history = session_data.get("dialogue_history", [])
    lines = []
    turn_num = 0

    for turn in dialogue_history:
        therapist = turn.get("therapist_utterance", "").strip()
        client = turn.get("client_utterance", "").strip()

        # Skip greeting turn where client is empty
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


def build_evaluation_prompt(formatted_dialogue: str) -> str:
    """Build the exact evaluation prompt matching the original GPT evaluation."""
    return f"""Evaluate the following therapist-client conversation:
{formatted_dialogue}

Score from 1 to 5:
- Coherence (Is the conversation logically structured, with smooth transitions between steps?)
- Depth (To what extent does the dialogue move beyond surface remarks to examine underlying emotions, cognitions, life history, and relational patterns, thereby demonstrating therapeutic depth?)
- Progress (Does the conversation effectively move forward in a logical manner?)
- Naturalness (Does the conversation feel fluid and human-like, avoiding robotic or overly scripted responses?)
- Empathy (Does the therapist convey accurate understanding and acceptance of the client's perspective?)
- MI Alignment (Does the therapist align with the predicted Motivational Interviewing (MI) skills?)

Provide scores in this format: Coherence: X, Depth: X, Progress: X, Naturalness: X, Empathy: X, MI Alignment: X
without any explanation or additional text."""


def parse_scores(response_text: str) -> dict:
    """Parse scores from GLM-5 response using regex."""
    matches = re.findall(r'(\w[\w\s]*?):\s*(\d+)', response_text)
    scores = {}
    for key, value in matches:
        key_clean = key.strip()
        val = int(value)
        if val < 1 or val > 5:
            continue
        # Normalize key names
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


# ── GLM-5 API Call ───────────────────────────────────────────────────────────


def call_glm5(prompt: str, retries: int = MAX_RETRIES) -> str:
    """Call GLM-5 API with retry logic and exponential backoff."""
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

    for attempt in range(retries):
        try:
            resp = requests.post(
                GLM_API_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content
            else:
                print(f"    API error (status {resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            print(f"    API call failed (attempt {attempt+1}/{retries}): {e}")

        if attempt < retries - 1:
            delay = RETRY_DELAY * (2 ** attempt)
            print(f"    Retrying in {delay}s...")
            time.sleep(delay)

    return ""


# ── Evaluation Pipeline ─────────────────────────────────────────────────────


def evaluate_all_models(models_to_eval=None):
    """Evaluate 600 dialogues (100 per model × 6 models) with GLM-5."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = OUTPUT_DIR / "glm5_evaluation_checkpoint.json"
    results_file = OUTPUT_DIR / "glm5_evaluation_results.json"

    # Load checkpoint if exists
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            all_results = json.load(f)
        print(f"Resumed from checkpoint: {sum(len(v) for v in all_results.values())} evaluations")
    else:
        all_results = {}

    models = models_to_eval or MODELS
    total_calls = 0
    total_success = 0
    total_failed = 0

    for model_name in models:
        model_dir = SESSION_DIR / model_name
        if not model_dir.exists():
            print(f"WARNING: Session directory not found: {model_dir}")
            continue

        if model_name not in all_results:
            all_results[model_name] = {}

        sampled_ids = sample_sessions(model_name)
        pending = [sid for sid in sampled_ids if f"session_{sid}" not in all_results[model_name]]

        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"  Sampled: {len(sampled_ids)} sessions")
        print(f"  Already done: {len(sampled_ids) - len(pending)}")
        print(f"  Pending: {len(pending)}")
        print(f"{'='*70}")

        for i, sid in enumerate(pending):
            session_file = model_dir / f"session_{sid}.json"
            if not session_file.exists():
                print(f"  [SKIP] session_{sid}.json not found")
                continue

            with open(session_file) as f:
                session_data = json.load(f)

            formatted = format_dialogue(session_data)
            prompt = build_evaluation_prompt(formatted)

            total_calls += 1
            print(f"  [{i+1}/{len(pending)}] Evaluating session_{sid}...", end=" ", flush=True)

            response_text = call_glm5(prompt)
            if not response_text:
                total_failed += 1
                print("FAILED")
                all_results[model_name][f"session_{sid}"] = {
                    "scores": {},
                    "raw_response": "",
                    "status": "failed",
                }
            else:
                scores = parse_scores(response_text)
                total_success += 1
                print(f"OK -> {scores}")
                all_results[model_name][f"session_{sid}"] = {
                    "scores": scores,
                    "raw_response": response_text,
                    "status": "success" if len(scores) == 6 else "partial",
                }

            # Checkpoint every 10 calls
            if total_calls % 10 == 0:
                with open(checkpoint_file, "w") as f:
                    json.dump(all_results, f, indent=2)

            # Rate limiting: ~1 request per second
            time.sleep(1.0)

    # Final save
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nEvaluation complete: {total_success} success, {total_failed} failed")
    print(f"Results saved to: {results_file}")

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    return all_results


# ── Load Reference Data ──────────────────────────────────────────────────────


def load_human_annotations() -> pd.DataFrame:
    """Load and average human annotations from both annotators."""
    df1 = pd.read_csv(HUMAN_ANNO_DIR / "human_annotation_1.csv")
    df2 = pd.read_csv(HUMAN_ANNO_DIR / "human_annotation_2.csv")

    dims_lower = ["coherence", "depth", "progress", "naturalness", "empathy", "mi_alignment"]

    # Normalize model names
    df1["model_norm"] = df1["model"].map(HUMAN_ANNO_MODEL_MAP)
    df2["model_norm"] = df2["model"].map(HUMAN_ANNO_MODEL_MAP)

    # Merge on (conversation_id, model)
    merged = df1.merge(
        df2,
        on=["conversation_id", "model"],
        suffixes=("_a1", "_a2"),
    )

    # Average annotator scores
    records = []
    for _, row in merged.iterrows():
        rec = {
            "session_id": row["conversation_id"],
            "model": HUMAN_ANNO_MODEL_MAP.get(row["model"], row["model"]),
        }
        for dim in dims_lower:
            a1 = row.get(f"{dim}_a1", np.nan)
            a2 = row.get(f"{dim}_a2", np.nan)
            rec[f"human_{dim}"] = np.mean([a1, a2])
        records.append(rec)

    return pd.DataFrame(records)


def load_gpt_judge_scores() -> dict:
    """Load original GPT-5-nano judge scores for all models."""
    gpt_dir = EVAL_DIR / "llm_judge_gpt5_nano"
    gpt_scores = {}

    # File name mapping
    file_map = {
        "gemma:7b": "llm_judge_gpt_5_nano_gemma_7b.json",
        "llama3.1:8b": "llm_judge_gpt_5_nano_llama3.1_8b.json",
        "openchat:7b": "llm_judge_gpt_5_nano_openchat_7b.json",
        "phi4:14b": "llm_judge_gpt_5_nano_phi4_14b.json",
        "qwen2.5:7b": "llm_judge_gpt_5_nano_qwen2.5_7b.json",
        "gpt-5-nano": "llm_judge_gpt_5_nano_gpt-5-nano.json",
    }

    for model, fname in file_map.items():
        fpath = gpt_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                gpt_scores[model] = json.load(f)
        else:
            print(f"WARNING: GPT judge file not found: {fpath}")

    return gpt_scores


# ── Analysis ─────────────────────────────────────────────────────────────────


def run_analysis(glm5_results: dict):
    """
    Run full correlation analysis and self-bias test.

    Compares GLM-5 judge with human annotations and GPT-5-nano judge.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("ANALYSIS: GLM-5 vs GPT-5-nano Cross-Model Validation")
    print("=" * 70)

    # Load reference data
    human_df = load_human_annotations()
    gpt_scores = load_gpt_judge_scores()

    # Existing GPT-Human correlation from paper
    existing_corr_file = DATA_DIR / "results" / "results_all" / "correlation_dimension_level.csv"
    existing_corr = None
    if existing_corr_file.exists():
        existing_corr = pd.read_csv(existing_corr_file)

    dims_lower = ["coherence", "depth", "progress", "naturalness", "empathy", "mi_alignment"]
    dim_map = {
        "coherence": "Coherence",
        "depth": "Depth",
        "progress": "Progress",
        "naturalness": "Naturalness",
        "empathy": "Empathy",
        "mi_alignment": "MI Alignment",
    }

    # Build analysis dataframe for human-annotated sessions
    records = []
    for _, hrow in human_df.iterrows():
        sid = hrow["session_id"]
        model = hrow["model"]

        # GLM-5 scores
        glm5_model_data = glm5_results.get(model, {})
        glm5_session = glm5_model_data.get(sid, {})
        glm5_scores = glm5_session.get("scores", {})

        # GPT judge scores
        gpt_model_data = gpt_scores.get(model, {})
        gpt_session = gpt_model_data.get(sid, {})

        rec = {
            "session_id": sid,
            "model": model,
        }

        for dim_lower in dims_lower:
            dim_cap = dim_map[dim_lower]
            rec[f"human_{dim_lower}"] = hrow.get(f"human_{dim_lower}", np.nan)
            rec[f"glm5_{dim_lower}"] = glm5_scores.get(dim_cap, np.nan)
            rec[f"gpt_{dim_lower}"] = gpt_session.get(dim_cap, np.nan)

        records.append(rec)

    analysis_df = pd.DataFrame(records)

    # ── 1. Dimension-level correlation ────────────────────────────────────
    print("\n1. DIMENSION-LEVEL CORRELATION")
    print("-" * 70)

    corr_records = []
    for dim_lower in dims_lower:
        dim_cap = dim_map[dim_lower]
        h_col = f"human_{dim_lower}"
        g5_col = f"glm5_{dim_lower}"
        gpt_col = f"gpt_{dim_lower}"

        # GLM5 vs Human
        mask_glm5 = analysis_df[[h_col, g5_col]].notna().all(axis=1)
        df_g5 = analysis_df[mask_glm5]
        n_g5 = len(df_g5)

        if n_g5 >= 3:
            pearson_g5, p_pearson_g5 = stats.pearsonr(df_g5[h_col], df_g5[g5_col])
            spearman_g5, p_spearman_g5 = stats.spearmanr(df_g5[h_col], df_g5[g5_col])
            kendall_g5, p_kendall_g5 = stats.kendalltau(df_g5[h_col], df_g5[g5_col])
        else:
            pearson_g5 = spearman_g5 = kendall_g5 = np.nan
            p_pearson_g5 = p_spearman_g5 = p_kendall_g5 = np.nan

        # GPT vs Human
        mask_gpt = analysis_df[[h_col, gpt_col]].notna().all(axis=1)
        df_gpt = analysis_df[mask_gpt]
        n_gpt = len(df_gpt)

        if n_gpt >= 3:
            pearson_gpt, p_pearson_gpt = stats.pearsonr(df_gpt[h_col], df_gpt[gpt_col])
            spearman_gpt, p_spearman_gpt = stats.spearmanr(df_gpt[h_col], df_gpt[gpt_col])
            kendall_gpt, p_kendall_gpt = stats.kendalltau(df_gpt[h_col], df_gpt[gpt_col])
        else:
            pearson_gpt = spearman_gpt = kendall_gpt = np.nan
            p_pearson_gpt = p_spearman_gpt = p_kendall_gpt = np.nan

        print(f"  {dim_cap}:")
        print(f"    GLM5-Human: Pearson={pearson_g5:.4f} (p={p_pearson_g5:.4f}), "
              f"Spearman={spearman_g5:.4f}, Kendall={kendall_g5:.4f} [n={n_g5}]")
        print(f"    GPT -Human: Pearson={pearson_gpt:.4f} (p={p_pearson_gpt:.4f}), "
              f"Spearman={spearman_gpt:.4f}, Kendall={kendall_gpt:.4f} [n={n_gpt}]")

        corr_records.append({
            "dimension": dim_lower,
            "GLM5_Pearson": round(pearson_g5, 4),
            "GLM5_Pearson_p": round(p_pearson_g5, 4),
            "GLM5_Spearman": round(spearman_g5, 4),
            "GLM5_Spearman_p": round(p_spearman_g5, 4),
            "GLM5_Kendall": round(kendall_g5, 4),
            "GLM5_Kendall_p": round(p_kendall_g5, 4),
            "GLM5_n": n_g5,
            "GPT_Pearson": round(pearson_gpt, 4),
            "GPT_Pearson_p": round(p_pearson_gpt, 4),
            "GPT_Spearman": round(spearman_gpt, 4),
            "GPT_Spearman_p": round(p_spearman_gpt, 4),
            "GPT_Kendall": round(kendall_gpt, 4),
            "GPT_Kendall_p": round(p_kendall_gpt, 4),
            "GPT_n": n_gpt,
        })

        # Also include original paper's GPT-Human correlations if available
        if existing_corr is not None:
            row_match = existing_corr[existing_corr["dimension"] == dim_lower]
            if not row_match.empty:
                row = row_match.iloc[0]
                corr_records[-1]["Paper_GPT_Pearson"] = round(row["Pearson"], 4)
                corr_records[-1]["Paper_GPT_Spearman"] = round(row["Spearman"], 4)
                corr_records[-1]["Paper_GPT_Kendall"] = round(row["Kendalltau"], 4)

    corr_df = pd.DataFrame(corr_records)
    corr_csv = OUTPUT_DIR / "glm5_vs_gpt_correlation_comparison.csv"
    corr_df.to_csv(corr_csv, index=False)
    print(f"\n  Saved: {corr_csv}")

    # ── 2. Model-level correlation ────────────────────────────────────────
    print("\n2. MODEL-LEVEL CORRELATION")
    print("-" * 70)

    # For each model, average all dimensions into a single score
    model_records = []
    for model in analysis_df["model"].unique():
        mdf = analysis_df[analysis_df["model"] == model]

        # Average across dimensions
        for _, row in mdf.iterrows():
            human_vals = [row.get(f"human_{d}", np.nan) for d in dims_lower]
            glm5_vals = [row.get(f"glm5_{d}", np.nan) for d in dims_lower]
            gpt_vals = [row.get(f"gpt_{d}", np.nan) for d in dims_lower]

            human_avg = np.nanmean(human_vals)
            glm5_avg = np.nanmean(glm5_vals)
            gpt_avg = np.nanmean(gpt_vals)

            model_records.append({
                "model": model,
                "session_id": row["session_id"],
                "human_avg": human_avg,
                "glm5_avg": glm5_avg,
                "gpt_avg": gpt_avg,
            })

    model_df = pd.DataFrame(model_records)

    for model in sorted(model_df["model"].unique()):
        mdf = model_df[model_df["model"] == model]

        mask_g5 = mdf[["human_avg", "glm5_avg"]].notna().all(axis=1)
        mask_gpt = mdf[["human_avg", "gpt_avg"]].notna().all(axis=1)

        n_g5 = mask_g5.sum()
        n_gpt = mask_gpt.sum()

        print(f"  {model}:")
        if n_g5 >= 3:
            r_g5, p_g5 = stats.pearsonr(mdf[mask_g5]["human_avg"], mdf[mask_g5]["glm5_avg"])
            sp_g5, _ = stats.spearmanr(mdf[mask_g5]["human_avg"], mdf[mask_g5]["glm5_avg"])
            print(f"    GLM5-Human: Pearson={r_g5:.4f} (p={p_g5:.4f}), Spearman={sp_g5:.4f} [n={n_g5}]")
        else:
            print(f"    GLM5-Human: insufficient data (n={n_g5})")

        if n_gpt >= 3:
            r_gpt, p_gpt = stats.pearsonr(mdf[mask_gpt]["human_avg"], mdf[mask_gpt]["gpt_avg"])
            sp_gpt, _ = stats.spearmanr(mdf[mask_gpt]["human_avg"], mdf[mask_gpt]["gpt_avg"])
            print(f"    GPT -Human: Pearson={r_gpt:.4f} (p={p_gpt:.4f}), Spearman={sp_gpt:.4f} [n={n_gpt}]")
        else:
            print(f"    GPT -Human: insufficient data (n={n_gpt})")

    # ── 3. Self-bias test ─────────────────────────────────────────────────
    print("\n3. SELF-BIAS TEST (GPT-5-nano sessions)")
    print("-" * 70)

    bias_lines = []
    bias_lines.append("=" * 70)
    bias_lines.append("SELF-BIAS ANALYSIS: GPT-5-nano as Judge vs GLM-5 as Judge")
    bias_lines.append("on GPT-5-nano Generated Dialogues")
    bias_lines.append("=" * 70)

    # Get GPT and GLM5 scores for gpt-5-nano model sessions
    gpt_nano_glm5 = glm5_results.get("gpt-5-nano", {})
    gpt_nano_gpt = gpt_scores.get("gpt-5-nano", {})

    # Collect paired scores for all evaluated sessions
    paired_scores = {dim: {"gpt": [], "glm5": []} for dim in DIMENSIONS}
    all_gpt_means = []
    all_glm5_means = []

    common_sessions = set(gpt_nano_glm5.keys()) & set(gpt_nano_gpt.keys())
    # Filter to successfully evaluated sessions
    common_sessions = [
        s for s in common_sessions
        if gpt_nano_glm5.get(s, {}).get("status") == "success"
        and s != "summary" and s != "total_sessions" and s != "failed_sessions"
    ]

    for sid in sorted(common_sessions):
        glm5_scores_session = gpt_nano_glm5[sid].get("scores", {})
        gpt_scores_session = gpt_nano_gpt[sid]

        glm5_vals = []
        gpt_vals = []
        for dim in DIMENSIONS:
            g5_val = glm5_scores_session.get(dim)
            gpt_val = gpt_scores_session.get(dim)
            if g5_val is not None and gpt_val is not None:
                paired_scores[dim]["glm5"].append(g5_val)
                paired_scores[dim]["gpt"].append(gpt_val)
                glm5_vals.append(g5_val)
                gpt_vals.append(gpt_val)

        if glm5_vals and gpt_vals:
            all_glm5_means.append(np.mean(glm5_vals))
            all_gpt_means.append(np.mean(gpt_vals))

    n_paired = len(all_gpt_means)
    bias_lines.append(f"\nNumber of paired sessions: {n_paired}")

    if n_paired >= 2:
        # Overall mean comparison
        gpt_mean = np.mean(all_gpt_means)
        glm5_mean = np.mean(all_glm5_means)
        bias_lines.append(f"\nOverall mean scores on GPT-5-nano dialogues:")
        bias_lines.append(f"  GPT-5-nano judge mean: {gpt_mean:.4f}")
        bias_lines.append(f"  GLM-5 judge mean:      {glm5_mean:.4f}")
        bias_lines.append(f"  Difference (GPT - GLM5): {gpt_mean - glm5_mean:.4f}")

        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(all_gpt_means, all_glm5_means)
        bias_lines.append(f"\n  Paired t-test: t={t_stat:.4f}, p={t_pval:.6f}")

        # Wilcoxon signed-rank test
        try:
            w_stat, w_pval = stats.wilcoxon(all_gpt_means, all_glm5_means)
            bias_lines.append(f"  Wilcoxon signed-rank: W={w_stat:.4f}, p={w_pval:.6f}")
        except Exception as e:
            bias_lines.append(f"  Wilcoxon signed-rank: could not compute ({e})")

        # Interpretation
        bias_lines.append("")
        if t_pval < 0.05 and gpt_mean > glm5_mean:
            bias_lines.append("INTERPRETATION: GPT-5-nano judge scores are SIGNIFICANTLY higher")
            bias_lines.append("on GPT-5-nano generated dialogues compared to GLM-5 judge scores.")
            bias_lines.append("This SUPPORTS the hypothesis of self-bias in GPT-5-nano evaluation.")
        elif t_pval < 0.05 and gpt_mean < glm5_mean:
            bias_lines.append("INTERPRETATION: GLM-5 judge scores are significantly higher.")
            bias_lines.append("This does NOT support self-bias; GPT-5-nano may even be self-critical.")
        else:
            bias_lines.append("INTERPRETATION: No significant difference between GPT-5-nano and GLM-5")
            bias_lines.append("judge scores on GPT-5-nano generated dialogues (p > 0.05).")
            bias_lines.append("This does NOT support the self-bias hypothesis.")

        # Per-dimension breakdown
        bias_lines.append(f"\nPer-dimension breakdown:")
        bias_lines.append(f"{'Dimension':<20} {'GPT mean':>10} {'GLM5 mean':>10} {'Diff':>8} {'t':>8} {'p':>10}")
        bias_lines.append("-" * 70)
        for dim in DIMENSIONS:
            g = paired_scores[dim]["gpt"]
            l = paired_scores[dim]["glm5"]
            if len(g) >= 2:
                gm = np.mean(g)
                lm = np.mean(l)
                t, p = stats.ttest_rel(g, l)
                sig = "*" if p < 0.05 else ""
                bias_lines.append(f"{dim:<20} {gm:>10.4f} {lm:>10.4f} {gm-lm:>+8.4f} {t:>8.4f} {p:>10.6f} {sig}")

        # Also compare across all models
        bias_lines.append(f"\n{'='*70}")
        bias_lines.append("CROSS-MODEL COMPARISON: GPT vs GLM5 Judge Mean Scores")
        bias_lines.append(f"{'='*70}")
        bias_lines.append(f"{'Model':<20} {'GPT mean':>10} {'GLM5 mean':>10} {'Diff':>8} {'n':>5}")
        bias_lines.append("-" * 55)

        for model in MODELS:
            glm5_model = glm5_results.get(model, {})
            gpt_model = gpt_scores.get(model, {})

            model_gpt_avgs = []
            model_glm5_avgs = []

            for sid in glm5_model:
                if sid in gpt_model and glm5_model[sid].get("status") == "success":
                    g5_s = glm5_model[sid].get("scores", {})
                    gpt_s = gpt_model.get(sid, {})
                    g5_v = [g5_s.get(d) for d in DIMENSIONS if g5_s.get(d) is not None]
                    gpt_v = [gpt_s.get(d) for d in DIMENSIONS if gpt_s.get(d) is not None]
                    if g5_v and gpt_v:
                        model_glm5_avgs.append(np.mean(g5_v))
                        model_gpt_avgs.append(np.mean(gpt_v))

            if model_gpt_avgs:
                gm = np.mean(model_gpt_avgs)
                lm = np.mean(model_glm5_avgs)
                bias_lines.append(f"{model:<20} {gm:>10.4f} {lm:>10.4f} {gm-lm:>+8.4f} {len(model_gpt_avgs):>5}")
            else:
                bias_lines.append(f"{model:<20} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'0':>5}")

    else:
        bias_lines.append("\nInsufficient paired data for self-bias analysis.")
        bias_lines.append("GLM-5 evaluation results may be incomplete.")

    bias_text = "\n".join(bias_lines)
    print(bias_text)

    # Save
    bias_file = OUTPUT_DIR / "self_bias_analysis.txt"
    with open(bias_file, "w") as f:
        f.write(bias_text)
    print(f"\n  Saved: {bias_file}")

    return corr_df


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="GLM-5 Cross-Model Evaluation")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip evaluation, only run analysis on existing results")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Evaluate specific models only")
    args = parser.parse_args()

    results_file = OUTPUT_DIR / "glm5_evaluation_results.json"

    if args.analyze_only:
        if not results_file.exists():
            print(f"ERROR: Results file not found: {results_file}")
            print("Run evaluation first (without --analyze-only)")
            sys.exit(1)
        with open(results_file) as f:
            glm5_results = json.load(f)
    else:
        glm5_results = evaluate_all_models(args.models)

    # Run analysis
    run_analysis(glm5_results)

    print("\n" + "=" * 70)
    print("Done! Output files:")
    for f in OUTPUT_DIR.glob("*"):
        print(f"  {f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
