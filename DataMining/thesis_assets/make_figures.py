"""
make_figures.py
Generate thesis-ready figures from the DataMining pipeline outputs.

Expected project structure (relative to DataMining/):
  results/reports/stage1/trial_quality.csv
  results/reports/stage1/participant_quality.csv
  results/reports/stage2/ta_per_trial.csv
  results/reports/stage3/stage3_with_labels.csv
  results/reports/stage2/AOI_Definitions_Overview.png  (optional)
  results/reports/stage2/AOI_Definitions_Detailed.png  (optional)

Outputs:
  thesis assets/figures/*.png
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[WARN] Missing: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _savefig(outpath: Path) -> None:
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[OK] Saved figure: {outpath}")


def fig_stage1_invalid_pct(trial_quality: pd.DataFrame, outdir: Path) -> None:
    if trial_quality is None or "invalid_pct" not in trial_quality.columns:
        print("[SKIP] Stage1 invalid_pct figure (missing data/column).")
        return

    data = trial_quality["invalid_pct"].dropna().astype(float)

    plt.figure(figsize=(7.5, 4.5))
    plt.hist(data, bins=30)
    plt.xlabel("Invalid sample proportion per trial (p_invalid)")
    plt.ylabel("Number of trials")
    plt.title("Stage 1: Distribution of Trial Invalid Proportions")
    _savefig(outdir / "fig_stage1_invalid_pct_distribution.png")


def fig_stage1_kept_excluded_by_part(trial_quality: pd.DataFrame, outdir: Path) -> None:
    if trial_quality is None:
        print("[SKIP] Stage1 kept/excluded by part (missing data).")
        return
    required = {"part", "excluded"}
    if not required.issubset(set(trial_quality.columns)):
        print("[SKIP] Stage1 kept/excluded by part (missing columns).")
        return

    df = trial_quality.copy()
    df["status"] = df["excluded"].map({True: "EXCLUDED", False: "KEPT"})

    pivot = (
        df.pivot_table(index="part", columns="status", values="question_id", aggfunc="count", fill_value=0)
        .reindex(columns=["KEPT", "EXCLUDED"], fill_value=0)
        .sort_index()
    )

    plt.figure(figsize=(7.5, 4.5))
    x = range(len(pivot.index))
    kept = pivot["KEPT"].values
    excl = pivot["EXCLUDED"].values

    plt.bar(x, kept, label="KEPT")
    plt.bar(x, excl, bottom=kept, label="EXCLUDED")
    plt.xticks(list(x), list(pivot.index), rotation=15)
    plt.ylabel("Number of trials")
    plt.title("Stage 1: Kept vs Excluded Trials by Condition")
    plt.legend()
    _savefig(outdir / "fig_stage1_kept_excluded_by_part.png")


def fig_ta_distribution(ta_df: pd.DataFrame, outdir: Path) -> None:
    if ta_df is None or "ta" not in ta_df.columns:
        print("[SKIP] ta distribution (missing data/column).")
        return

    valid = ta_df["ta"].dropna().astype(float)
    plt.figure(figsize=(7.5, 4.5))
    plt.hist(valid, bins=30)
    plt.xlabel("t_a (seconds)")
    plt.ylabel("Number of trials")
    plt.title("Stage 2: Distribution of First Stable Fixation Onset Time (t_a)")
    _savefig(outdir / "fig_stage2_ta_distribution.png")


def fig_label_distribution(stage3: pd.DataFrame, outdir: Path) -> None:
    if stage3 is None or "randomness_label" not in stage3.columns:
        print("[SKIP] label distribution (missing data/column).")
        return

    counts = stage3["randomness_label"].value_counts()
    order = ["NOT_RANDOM", "RANDOM", "INVALID"]
    counts = counts.reindex(order).fillna(0)

    plt.figure(figsize=(7.0, 4.5))
    plt.bar(range(len(counts)), counts.values)
    plt.xticks(range(len(counts)), counts.index)
    plt.ylabel("Number of trials")
    plt.title("Stage 3: Randomness Label Distribution (All Trials)")
    _savefig(outdir / "fig_stage3_label_distribution.png")


def fig_label_by_part(stage3: pd.DataFrame, outdir: Path) -> None:
    if stage3 is None:
        print("[SKIP] label by part (missing data).")
        return
    required = {"part", "randomness_label"}
    if not required.issubset(set(stage3.columns)):
        print("[SKIP] label by part (missing columns).")
        return

    df = stage3.copy()
    order_labels = ["NOT_RANDOM", "RANDOM", "INVALID"]
    order_parts = ["No-Timer", "Timer-Correct", "Timer-No-Correct"]

    pivot = (
        df.pivot_table(index="part", columns="randomness_label", values="question_id", aggfunc="count", fill_value=0)
        .reindex(index=order_parts)
        .reindex(columns=order_labels, fill_value=0)
        .fillna(0)
    )

    plt.figure(figsize=(8.5, 4.8))
    x = range(len(pivot.index))
    bottom = [0] * len(pivot.index)

    for lab in order_labels:
        vals = pivot[lab].values
        plt.bar(x, vals, bottom=bottom, label=lab)
        bottom = [b + v for b, v in zip(bottom, vals)]

    plt.xticks(list(x), list(pivot.index), rotation=15)
    plt.ylabel("Number of trials")
    plt.title("Stage 3: Label Distribution by Condition")
    plt.legend()
    _savefig(outdir / "fig_stage3_label_by_condition.png")


def fig_tanswer_distribution(stage3: pd.DataFrame, outdir: Path) -> None:
    if stage3 is None or "t_answer" not in stage3.columns:
        print("[SKIP] t_answer distribution (missing data/column).")
        return

    valid = stage3[stage3["randomness_label"].isin(["NOT_RANDOM", "RANDOM"])].copy()
    if valid.empty:
        print("[SKIP] t_answer distribution (no valid trials).")
        return

    vals = valid["t_answer"].dropna().astype(float)
    plt.figure(figsize=(7.5, 4.5))
    plt.hist(vals, bins=30)
    plt.xlabel("t_answer = t_end - t_a (seconds)")
    plt.ylabel("Number of trials")
    plt.title("Stage 3: Distribution of Post-Engagement Commitment Time (t_answer)")
    _savefig(outdir / "fig_stage3_tanswer_distribution.png")


def maybe_copy_aoi_images(stage2_dir: Path, outdir: Path) -> None:
    # Optional: copy AOI definition images generated by stage2
    for name in ["AOI_Definitions_Overview.png", "AOI_Definitions_Detailed.png"]:
        src = stage2_dir / name
        if src.exists():
            dst = outdir / name
            try:
                dst.write_bytes(src.read_bytes())
                print(f"[OK] Copied: {src} -> {dst}")
            except Exception as e:
                print(f"[WARN] Could not copy {src}: {e}")


def main() -> int:
    # This script is placed in: DataMining/thesis assets/make_figures.py
    # So DataMining is two levels up.
    here = Path(__file__).resolve()
    datamining_dir = here.parents[1]  # .../DataMining
    reports_dir = datamining_dir / "results" / "reports"

    stage1_trial = reports_dir / "stage1" / "trial_quality.csv"
    stage2_ta = reports_dir / "stage2" / "ta_per_trial.csv"
    stage3_labels = reports_dir / "stage3" / "stage3_with_labels.csv"

    figs_dir = datamining_dir / "thesis assets" / "figures"
    _ensure_dir(figs_dir)

    trial_quality = _safe_read_csv(stage1_trial)
    ta_df = _safe_read_csv(stage2_ta)
    stage3 = _safe_read_csv(stage3_labels)

    # Optional copy of AOI drawings (if generated by stage2)
    maybe_copy_aoi_images(reports_dir / "stage2", figs_dir)

    # Figures
    fig_stage1_invalid_pct(trial_quality, figs_dir)
    fig_stage1_kept_excluded_by_part(trial_quality, figs_dir)
    fig_ta_distribution(ta_df, figs_dir)
    fig_label_distribution(stage3, figs_dir)
    fig_label_by_part(stage3, figs_dir)
    fig_tanswer_distribution(stage3, figs_dir)

    print("\n[DONE] All possible figures generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())