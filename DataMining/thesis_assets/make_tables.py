"""
make_tables.py
Generate LaTeX-ready tables from the DataMining pipeline outputs.

Outputs:
  thesis assets/tables/*.tex

You can include them in LaTeX like:
  \\input{tables/table_dataset_overview}
  \\input{tables/table_stage1_quality}
  \\input{tables/table_ta_summary}
  \\input{tables/table_label_distribution}
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from configparser import ConfigParser


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


def _write_tex(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    print(f"[OK] Saved table: {path}")


def _latex_escape(s: str) -> str:
    # minimal escape for LaTeX
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def df_to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    # Simple, journal-friendly table (no booktabs dependency assumed)
    # If you already use booktabs, tell me and I’ll switch to \\toprule/\\midrule/\\bottomrule.
    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)

    header = " & ".join([_latex_escape(c) for c in cols]) + " \\\\ \\hline\n"
    rows = ""
    for _, row in df.iterrows():
        rows += " & ".join(_latex_escape(v) for v in row.values) + " \\\\ \\hline\n"

    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{{_latex_escape(caption)}}}
\\label{{{_latex_escape(label)}}}
\\begin{{tabular}}{{{align}}}
\\hline
{header}{rows}\\end{{tabular}}
\\end{{table}}
"""
    return tex


def load_config(config_path: Path) -> ConfigParser | None:
    if not config_path.exists():
        print(f"[WARN] Config not found: {config_path}")
        return None
    cfg = ConfigParser()
    cfg.read(config_path)
    return cfg


def table_dataset_overview(trial_quality: pd.DataFrame | None,
                           participant_quality: pd.DataFrame | None,
                           ta_df: pd.DataFrame | None,
                           stage3: pd.DataFrame | None,
                           cfg: ConfigParser | None) -> pd.DataFrame:

    # defaults if something is missing
    total_trials = None
    kept_trials = None
    excluded_trials = None
    total_participants = None
    excluded_participants = None
    ta_rate = None
    valid_label_trials = None

    if trial_quality is not None:
        total_trials = int(len(trial_quality))
        if "excluded" in trial_quality.columns:
            excluded_trials = int(trial_quality["excluded"].sum())
            kept_trials = int(total_trials - excluded_trials)

    if participant_quality is not None and "participant_excluded" in participant_quality.columns:
        total_participants = int(len(participant_quality))
        excluded_participants = int(participant_quality["participant_excluded"].sum())

    if ta_df is not None and "ta" in ta_df.columns:
        ta_rate = float(ta_df["ta"].notna().mean() * 100.0)

    if stage3 is not None and "randomness_label" in stage3.columns:
        valid_label_trials = int(stage3["randomness_label"].isin(["RANDOM", "NOT_RANDOM"]).sum())

    sampling_rate = None
    if cfg is not None:
        try:
            sampling_rate = cfg.getint("Analysis", "gaze_sampling_rate_hz", fallback=None)
        except Exception:
            sampling_rate = None

    rows = [
        ["Total participants (main study)", total_participants if total_participants is not None else "—"],
        ["Excluded participants (Stage 1 rule)", excluded_participants if excluded_participants is not None else "—"],
        ["Total trials (participant × question)", total_trials if total_trials is not None else "—"],
        ["Kept trials after Stage 1", kept_trials if kept_trials is not None else "—"],
        ["Excluded trials after Stage 1", excluded_trials if excluded_trials is not None else "—"],
        ["Trials with detected $t_a$ (rate)", f"{ta_rate:.1f}\\%" if ta_rate is not None else "—"],
        ["Trials with valid label (RANDOM/NOT\\_RANDOM)", valid_label_trials if valid_label_trials is not None else "—"],
        ["Eye-tracker sampling rate", f"{sampling_rate} Hz" if sampling_rate is not None else "150 Hz"],
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def table_stage1_quality(trial_quality: pd.DataFrame | None,
                         participant_quality: pd.DataFrame | None,
                         cfg: ConfigParser | None) -> pd.DataFrame:
    tau_trial = 0.30
    tau_participant = 0.50
    if cfg is not None:
        tau_trial = cfg.getfloat("Analysis", "stage1_invalid_pct_threshold", fallback=0.30)
        tau_participant = cfg.getfloat("Analysis", "stage1_participant_exclusion_threshold", fallback=0.50)

    # Compute summary from available dfs
    total_trials = kept_trials = excluded_trials = "—"
    if trial_quality is not None and "excluded" in trial_quality.columns:
        total_trials = int(len(trial_quality))
        excluded_trials = int(trial_quality["excluded"].sum())
        kept_trials = int(total_trials - excluded_trials)

    total_participants = kept_participants = excluded_participants = "—"
    if participant_quality is not None and "participant_excluded" in participant_quality.columns:
        total_participants = int(len(participant_quality))
        excluded_participants = int(participant_quality["participant_excluded"].sum())
        kept_participants = int(total_participants - excluded_participants)

    rows = [
        ["Trial invalid threshold $\\tau_{trial}$", f"{tau_trial:.2f}"],
        ["Participant exclusion threshold $\\tau_{participant}$", f"{tau_participant:.2f}"],
        ["Total trials evaluated", total_trials],
        ["Kept trials", kept_trials],
        ["Excluded trials", excluded_trials],
        ["Total participants evaluated", total_participants],
        ["Kept participants", kept_participants],
        ["Excluded participants", excluded_participants],
    ]
    return pd.DataFrame(rows, columns=["Stage 1 Summary", "Value"])


def table_ta_summary(ta_df: pd.DataFrame | None,
                     cfg: ConfigParser | None) -> pd.DataFrame:
    window_ms = 1000
    theta = 0.85
    fs = 150
    if cfg is not None:
        window_ms = cfg.getint("STAGE2_TA", "ta_window_ms", fallback=1000)
        theta = cfg.getfloat("STAGE2_TA", "ta_answer_coverage_threshold", fallback=0.85)
        fs = cfg.getint("Analysis", "gaze_sampling_rate_hz", fallback=150)

    if ta_df is None or "ta" not in ta_df.columns:
        rows = [
            ["Stability window W (ms)", window_ms],
            ["Coverage threshold $\\theta$", theta],
            ["Sampling rate $f_s$ (Hz)", fs],
            ["Trials with $t_a$", "—"],
            ["$t_a$ mean (s)", "—"],
            ["$t_a$ median (s)", "—"],
            ["$t_a$ min (s)", "—"],
            ["$t_a$ max (s)", "—"],
        ]
        return pd.DataFrame(rows, columns=["Stage 2 Summary", "Value"])

    valid = ta_df["ta"].dropna().astype(float)
    rows = [
        ["Stability window W (ms)", window_ms],
        ["Coverage threshold $\\theta$", f"{theta:.2f}"],
        ["Sampling rate $f_s$ (Hz)", fs],
        ["Trials with $t_a$", f"{len(valid)}/{len(ta_df)} ({valid.shape[0]/len(ta_df)*100:.1f}\\%)"],
        ["$t_a$ mean (s)", f"{valid.mean():.3f}"],
        ["$t_a$ median (s)", f"{valid.median():.3f}"],
        ["$t_a$ min (s)", f"{valid.min():.3f}"],
        ["$t_a$ max (s)", f"{valid.max():.3f}"],
    ]
    return pd.DataFrame(rows, columns=["Stage 2 Summary", "Value"])


def table_label_distribution(stage3: pd.DataFrame | None,
                             cfg: ConfigParser | None) -> pd.DataFrame:
    percentile = 25.0
    if cfg is not None:
        percentile = cfg.getfloat("Analysis", "stage3_threshold_percentile", fallback=25.0)

    if stage3 is None or "randomness_label" not in stage3.columns:
        rows = [
            ["Percentile threshold (P)", f"{percentile:.0f}"],
            ["NOT\\_RANDOM", "—"],
            ["RANDOM", "—"],
            ["INVALID", "—"],
        ]
        return pd.DataFrame(rows, columns=["Stage 3 Summary", "Value"])

    counts = stage3["randomness_label"].value_counts()
    nr = int(counts.get("NOT_RANDOM", 0))
    r = int(counts.get("RANDOM", 0))
    inv = int(counts.get("INVALID", 0))
    total = int(len(stage3))
    valid = nr + r

    rows = [
        ["Percentile threshold (P)", f"{percentile:.0f}"],
        ["Total trials", total],
        ["Valid trials (RANDOM/NOT\\_RANDOM)", valid],
        ["NOT\\_RANDOM", f"{nr} ({(nr/valid*100 if valid else 0):.1f}\\% of valid)"],
        ["RANDOM", f"{r} ({(r/valid*100 if valid else 0):.1f}\\% of valid)"],
        ["INVALID", f"{inv} ({(inv/total*100 if total else 0):.1f}\\% of all)"],
    ]
    return pd.DataFrame(rows, columns=["Stage 3 Summary", "Value"])


def table_label_by_condition(stage3: pd.DataFrame | None) -> pd.DataFrame | None:
    if stage3 is None:
        return None
    required = {"part", "randomness_label"}
    if not required.issubset(set(stage3.columns)):
        return None

    order_parts = ["No-Timer", "Timer-Correct", "Timer-No-Correct"]
    order_labels = ["NOT_RANDOM", "RANDOM", "INVALID"]

    pivot = (
        stage3.pivot_table(index="part", columns="randomness_label", values="question_id", aggfunc="count", fill_value=0)
        .reindex(index=order_parts)
        .reindex(columns=order_labels, fill_value=0)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    pivot.rename(columns={"part": "Condition"}, inplace=True)
    return pivot


def main() -> int:
    # This script is placed in: DataMining/thesis assets/make_tables.py
    here = Path(__file__).resolve()
    datamining_dir = here.parents[1]  # .../DataMining
    reports_dir = datamining_dir / "results" / "reports"
    tables_dir = datamining_dir / "thesis assets" / "tables"
    _ensure_dir(tables_dir)

    cfg = load_config(datamining_dir / "config.ini")

    trial_quality = _safe_read_csv(reports_dir / "stage1" / "trial_quality.csv")
    participant_quality = _safe_read_csv(reports_dir / "stage1" / "participant_quality.csv")
    ta_df = _safe_read_csv(reports_dir / "stage2" / "ta_per_trial.csv")
    stage3 = _safe_read_csv(reports_dir / "stage3" / "stage3_with_labels.csv")

    # --- Table 1: dataset overview ---
    df1 = table_dataset_overview(trial_quality, participant_quality, ta_df, stage3, cfg)
    _write_tex(
        tables_dir / "table_dataset_overview.tex",
        df_to_latex_table(df1, "Dataset overview and pipeline yield.", "tab:dataset-overview"),
    )

    # --- Table 2: Stage 1 summary ---
    df2 = table_stage1_quality(trial_quality, participant_quality, cfg)
    _write_tex(
        tables_dir / "table_stage1_quality.tex",
        df_to_latex_table(df2, "Stage 1 quality filtering summary.", "tab:stage1-quality"),
    )

    # --- Table 3: Stage 2 ta summary ---
    df3 = table_ta_summary(ta_df, cfg)
    _write_tex(
        tables_dir / "table_ta_summary.tex",
        df_to_latex_table(df3, "Stage 2 stable-fixation onset ($t_a$) summary.", "tab:ta-summary"),
    )

    # --- Table 4: Stage 3 label distribution ---
    df4 = table_label_distribution(stage3, cfg)
    _write_tex(
        tables_dir / "table_label_distribution.tex",
        df_to_latex_table(df4, "Stage 3 randomness label distribution.", "tab:label-distribution"),
    )

    # --- Table 5: Label distribution by condition (optional but very useful for Results) ---
    df5 = table_label_by_condition(stage3)
    if df5 is not None:
        _write_tex(
            tables_dir / "table_label_by_condition.tex",
            df_to_latex_table(df5, "Randomness labels stratified by experimental condition.", "tab:label-by-condition"),
        )
    else:
        print("[SKIP] table_label_by_condition (missing columns).")

    print("\n[DONE] All possible tables generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())