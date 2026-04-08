#!/usr/bin/env python3
"""
Questionnaire Heterogeneity and Representativity Analysis

Analyzes the distributional properties of 1,000 generated client questionnaire
profiles to address reviewer concerns about population coverage.

Outputs:
  - questionnaire_heterogeneity_report.json
  - questionnaire_heterogeneity_summary.txt
  - figures/demographic_histogram.pdf
  - figures/severity_heatmap.pdf
  - figures/comorbidity_histogram.pdf
  - figures/total_score_histogram.pdf
"""

import json
import glob
import os
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "results" / "questionnaires"
OUTPUT_DIR = BASE_DIR / "data" / "results"
FIGURE_DIR = OUTPUT_DIR / "figures"

DOMAIN_ITEMS = {
    "Depression": [0, 1],
    "Anger": [2, 3],
    "Mania": [4, 5],
    "Anxiety": [6, 7],
    "Somatic symptoms": [8, 9, 10],
    "Suicidal ideation": [11],
    "Psychosis": [12, 13],
    "Sleep problems": [14],
    "Memory": [15],
    "Repetitive thoughts and behaviors": [16, 17],
    "Dissociation": [18, 19],
    "Personality functioning": [20, 21],
    "Substance use": [22],
}

DOMAIN_ORDER = list(DOMAIN_ITEMS.keys())
SEVERITY_LABELS = ["None (0)", "Slight (1)", "Mild (2)", "Moderate (3)", "Severe (4)"]
N_ITEMS = 23  # expected number of questionnaire items

# ── Data Loading ─────────────────────────────────────────────────────────────


def load_profiles():
    """Load all questionnaire profiles, computing domain scores from raw items."""
    files = sorted(glob.glob(str(DATA_DIR / "questionnaire_user*.json")))
    profiles = []
    skipped = 0

    for fpath in files:
        with open(fpath) as f:
            raw = json.load(f)

        user_info = raw.get("user_info", "")
        level1 = raw.get("questionnaire", {}).get("level1", {})
        user_resp = level1.get("user_response", {})
        scores = user_resp.get("scores")

        # Skip profiles without scores
        if scores is None:
            skipped += 1
            continue

        # Parse identity and age from user_info: "{age}_{identity}_{domain_hint}"
        parts = user_info.split("_", 2)  # split into at most 3 parts
        try:
            age = int(parts[0])
        except (ValueError, IndexError):
            skipped += 1
            continue
        identity = parts[1].strip().lower() if len(parts) > 1 else "unknown"

        # Normalize identity type
        identity_map = {
            "adult": "Adult",
            "child": "Child 11-17",
            "child 11-17": "Child 11-17",
            "parent": "Parent of Child 6-17",
            "parent of child 6-17": "Parent of Child 6-17",
        }
        identity_type = identity_map.get(identity, identity.title())

        # Compute domain severity from raw scores (max of domain items)
        # Use first N_ITEMS items, pad with 0 if shorter
        padded = (list(scores) + [0] * N_ITEMS)[:N_ITEMS]
        domain_scores = {}
        for domain, items in DOMAIN_ITEMS.items():
            domain_scores[domain] = max(padded[i] for i in items)

        # Total score: sum of first 23 items
        total_score = sum(padded)

        profiles.append({
            "age": age,
            "identity_type": identity_type,
            "scores_raw": padded,
            "domain_scores": domain_scores,
            "total_score": total_score,
        })

    print(f"Loaded {len(profiles)} profiles ({skipped} skipped)")
    return profiles


# ── Analysis Functions ───────────────────────────────────────────────────────


def demographic_analysis(profiles):
    """1. Demographic distribution analysis."""
    ages = [p["age"] for p in profiles]
    identities = [p["identity_type"] for p in profiles]

    id_counts = Counter(identities)
    total = len(profiles)

    result = {
        "identity_type_counts": {k: v for k, v in sorted(id_counts.items())},
        "identity_type_pcts": {
            k: round(100.0 * v / total, 1)
            for k, v in sorted(id_counts.items())
        },
        "age_mean": round(float(np.mean(ages)), 1),
        "age_std": round(float(np.std(ages)), 1),
        "age_min": int(np.min(ages)),
        "age_max": int(np.max(ages)),
        "age_median": round(float(np.median(ages)), 1),
        "age_q25": round(float(np.percentile(ages, 25)), 1),
        "age_q75": round(float(np.percentile(ages, 75)), 1),
    }

    # Plot age histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ages, bins=10, edgecolor="black", alpha=0.75, color="#4C72B0")
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Age Distribution of 1,000 Client Profiles", fontsize=13)
    ax.axvline(np.mean(ages), color="red", linestyle="--", linewidth=1.2,
               label=f"Mean = {np.mean(ages):.1f}")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "demographic_histogram.pdf", dpi=150)
    plt.close(fig)
    print("  Saved demographic_histogram.pdf")

    return result


def severity_distribution(profiles):
    """2. Symptom severity distribution (13 domains × 5 severity levels)."""
    # Build matrix: domain -> severity_level -> count
    n = len(profiles)
    heatmap = {}  # domain -> [count_0, count_1, ..., count_4]

    for domain in DOMAIN_ORDER:
        counts = [0] * 5
        for p in profiles:
            sev = p["domain_scores"][domain]
            sev = min(max(sev, 0), 4)  # clamp to [0,4]
            counts[sev] += 1
        heatmap[domain] = {
            "counts": counts,
            "percentages": [round(100.0 * c / n, 1) for c in counts],
        }

    # Build matrix for heatmap plot
    pct_matrix = np.array([heatmap[d]["percentages"] for d in DOMAIN_ORDER])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pct_matrix,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        xticklabels=SEVERITY_LABELS,
        yticklabels=DOMAIN_ORDER,
        cbar_kws={"label": "Percentage of Profiles (%)"},
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Symptom Severity Distribution Across 13 DSM-5 Domains", fontsize=12)
    ax.set_xlabel("Severity Level", fontsize=11)
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "severity_heatmap.pdf", dpi=150)
    plt.close(fig)
    print("  Saved severity_heatmap.pdf")

    return heatmap


def comorbidity_analysis(profiles):
    """3. Comorbidity analysis (elevated = domain max score >= 2)."""
    elevated_counts = []  # number of elevated domains per profile
    pair_counter = Counter()

    for p in profiles:
        elevated_domains = [
            d for d in DOMAIN_ORDER if p["domain_scores"][d] >= 2
        ]
        elevated_counts.append(len(elevated_domains))

        # Count co-elevated pairs
        for d1, d2 in combinations(elevated_domains, 2):
            pair_counter[(d1, d2)] += 1

    ec = np.array(elevated_counts)
    dist = Counter(elevated_counts)

    result = {
        "elevated_count_mean": round(float(np.mean(ec)), 2),
        "elevated_count_std": round(float(np.std(ec)), 2),
        "elevated_count_min": int(np.min(ec)),
        "elevated_count_max": int(np.max(ec)),
        "elevated_count_median": round(float(np.median(ec)), 1),
        "elevated_count_distribution": {str(k): v for k, v in sorted(dist.items())},
        "top5_co_elevated_pairs": [
            {"pair": list(pair), "count": count, "pct": round(100.0 * count / len(profiles), 1)}
            for pair, count in pair_counter.most_common(5)
        ],
    }

    # Plot comorbidity histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = range(0, 14 + 2)  # 0 to 14
    ax.hist(elevated_counts, bins=bins, edgecolor="black", alpha=0.75,
            color="#55A868", align="left")
    ax.set_xlabel("Number of Elevated Domains (score ≥ 2)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Comorbidity Distribution: Elevated Domain Count per Profile",
                 fontsize=12)
    ax.axvline(np.mean(ec), color="red", linestyle="--", linewidth=1.2,
               label=f"Mean = {np.mean(ec):.1f}")
    ax.legend(fontsize=10)
    ax.set_xticks(range(0, 14))
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "comorbidity_histogram.pdf", dpi=150)
    plt.close(fig)
    print("  Saved comorbidity_histogram.pdf")

    return result


def total_score_analysis(profiles):
    """4. Total score distribution."""
    totals = np.array([p["total_score"] for p in profiles])

    result = {
        "mean": round(float(np.mean(totals)), 2),
        "std": round(float(np.std(totals)), 2),
        "min": int(np.min(totals)),
        "max": int(np.max(totals)),
        "q25": round(float(np.percentile(totals, 25)), 1),
        "median": round(float(np.median(totals)), 1),
        "q75": round(float(np.percentile(totals, 75)), 1),
    }

    # Plot total score histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(totals, bins=20, edgecolor="black", alpha=0.75, color="#C44E52")
    ax.set_xlabel("Total Score (sum of 23 items, range 0–92)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Total Questionnaire Score Distribution", fontsize=13)
    ax.axvline(np.mean(totals), color="navy", linestyle="--", linewidth=1.2,
               label=f"Mean = {np.mean(totals):.1f}")
    ax.axvline(np.median(totals), color="green", linestyle=":", linewidth=1.2,
               label=f"Median = {np.median(totals):.1f}")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "total_score_histogram.pdf", dpi=150)
    plt.close(fig)
    print("  Saved total_score_histogram.pdf")

    return result


def build_summary(demo, severity, comorbidity, total, n_profiles):
    """5. Build summary statistics table for rebuttal text."""
    # Identity type string
    id_parts = []
    for k, v in sorted(demo["identity_type_pcts"].items()):
        id_parts.append(f"{k}: {v}%")
    n_types = len(demo["identity_type_counts"])
    id_str = f"{n_types} ({', '.join(id_parts)})"

    # Count domains with full severity coverage (all 5 levels represented)
    domains_covered = 0
    for domain in DOMAIN_ORDER:
        counts = severity[domain]["counts"]
        if any(c > 0 for c in counts):
            domains_covered += 1

    lines = [
        "=" * 72,
        "QUESTIONNAIRE HETEROGENEITY & REPRESENTATIVITY SUMMARY",
        "=" * 72,
        "",
        "| Metric                              | Value                          |",
        "|--------------------------------------|--------------------------------|",
        f"| Profiles                             | {n_profiles:,}                          |",
        f"| Identity types                       | {id_str:<30} |",
        f"| Age range                            | {demo['age_min']} – {demo['age_max']} (mean {demo['age_mean']}, SD {demo['age_std']}) |",
        f"| Domains covered                      | {domains_covered}/13                        |",
        f"| Avg elevated domains per profile     | {comorbidity['elevated_count_mean']} (SD {comorbidity['elevated_count_std']})              |",
        f"| Total score range                    | {total['min']} – {total['max']} (mean {total['mean']}, SD {total['std']}) |",
        "",
        "-" * 72,
        "DETAILED RESULTS",
        "-" * 72,
        "",
        "1. DEMOGRAPHIC DISTRIBUTION",
        f"   Identity types: {id_str}",
        f"   Age: mean={demo['age_mean']}, SD={demo['age_std']}, "
        f"range=[{demo['age_min']}, {demo['age_max']}], "
        f"median={demo['age_median']}, IQR=[{demo['age_q25']}, {demo['age_q75']}]",
        "",
        "2. SYMPTOM SEVERITY DISTRIBUTION (% of profiles at each level)",
        f"   {'Domain':<38} {'None':>6} {'Slight':>7} {'Mild':>6} {'Mod':>6} {'Severe':>7}",
        "   " + "-" * 66,
    ]

    for domain in DOMAIN_ORDER:
        pcts = severity[domain]["percentages"]
        lines.append(
            f"   {domain:<38} {pcts[0]:>5.1f}% {pcts[1]:>6.1f}% {pcts[2]:>5.1f}% "
            f"{pcts[3]:>5.1f}% {pcts[4]:>6.1f}%"
        )

    lines += [
        "",
        "3. COMORBIDITY ANALYSIS (elevated = domain score >= 2)",
        f"   Mean elevated domains: {comorbidity['elevated_count_mean']} "
        f"(SD {comorbidity['elevated_count_std']})",
        f"   Range: {comorbidity['elevated_count_min']} – "
        f"{comorbidity['elevated_count_max']}",
        "",
        "   Top-5 co-elevated domain pairs:",
    ]
    for i, pair_info in enumerate(comorbidity["top5_co_elevated_pairs"], 1):
        lines.append(
            f"   {i}. {pair_info['pair'][0]} & {pair_info['pair'][1]}: "
            f"{pair_info['count']} profiles ({pair_info['pct']}%)"
        )

    lines += [
        "",
        "4. TOTAL SCORE DISTRIBUTION",
        f"   Mean={total['mean']}, SD={total['std']}, "
        f"Range=[{total['min']}, {total['max']}]",
        f"   Quartiles: Q1={total['q25']}, Median={total['median']}, "
        f"Q3={total['q75']}",
        "",
        "=" * 72,
    ]

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    # Ensure output directories exist
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading profiles...")
    profiles = load_profiles()
    n = len(profiles)
    print(f"Analyzing {n} profiles\n")

    print("[1/5] Demographic distribution...")
    demo = demographic_analysis(profiles)

    print("[2/5] Severity distribution...")
    severity = severity_distribution(profiles)

    print("[3/5] Comorbidity analysis...")
    comorbidity = comorbidity_analysis(profiles)

    print("[4/5] Total score distribution...")
    total = total_score_analysis(profiles)

    print("[5/5] Building summary...")
    summary_text = build_summary(demo, severity, comorbidity, total, n)

    # ── Save JSON report ─────────────────────────────────────────────────
    report = {
        "n_profiles": n,
        "demographic": demo,
        "severity_distribution": severity,
        "comorbidity": comorbidity,
        "total_score": total,
    }

    report_path = OUTPUT_DIR / "questionnaire_heterogeneity_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {report_path}")

    # ── Save summary text ────────────────────────────────────────────────
    summary_path = OUTPUT_DIR / "questionnaire_heterogeneity_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Saved {summary_path}")

    # Print summary to stdout
    print("\n" + summary_text)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
