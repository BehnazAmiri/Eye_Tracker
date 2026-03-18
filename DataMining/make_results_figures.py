import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG - adjust paths/columns once
# =========================
BASE_DIR = r"D:\MasterThesis\MasterThesis"
STAGE3_CSV = os.path.join(BASE_DIR, r"DataMining\results\reports\stage3\stage3_with_labels.csv")

# Optional: if you have these exports; otherwise figures that depend on them will be skipped
STAGE1_TRIAL_QC_CSV = os.path.join(BASE_DIR, r"DataMining\results\reports\stage1\trial_quality.csv")
STAGE2_TA_CSV = os.path.join(BASE_DIR, r"DataMining\results\reports\stage2\ta_results.csv")

OUT_DIR = os.path.join(BASE_DIR, r"MasterThesis\figures\results")
os.makedirs(OUT_DIR, exist_ok=True)

# Column name assumptions (edit if needed)
COL_PART = "part"
COL_LABEL = "randomness_label"          # values: RANDOM / NOT_RANDOM / INVALID
COL_TA = "ta"                           # numeric seconds, or NaN if not detected
COL_INVALID_RATIO = "invalid_ratio"     # 0..1 trial invalid fraction


def safe_read_csv(path: str):
    if not os.path.exists(path):
        print(f"[WARN] Missing file, skipping: {path}")
        return None
    return pd.read_csv(path)


def plot_stage1_invalid_distribution(df_qc: pd.DataFrame):
    if df_qc is None or COL_INVALID_RATIO not in df_qc.columns:
        print("[WARN] Stage1 invalid distribution skipped (file/column not available).")
        return

    x = df_qc[COL_INVALID_RATIO].dropna().values

    plt.figure()
    plt.hist(x, bins=30)
    plt.xlabel("Invalid sample ratio per trial")
    plt.ylabel("Number of trials")
    plt.title("Stage 1: Trial-level invalid ratio distribution")

    out_path = os.path.join(OUT_DIR, "invalid_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out_path}")


def plot_stage2_ta_distribution(df_ta: pd.DataFrame):
    if df_ta is None or COL_TA not in df_ta.columns:
        print("[WARN] Stage2 ta distribution skipped (file/column not available).")
        return

    x = df_ta[COL_TA].dropna().values
    if len(x) == 0:
        print("[WARN] No ta values found; skipping ta distribution.")
        return

    plt.figure()
    plt.hist(x, bins=30)
    plt.xlabel("t_a (seconds)")
    plt.ylabel("Number of trials")
    plt.title("Stage 2: Distribution of stable Answer-AOI onset (t_a)")

    out_path = os.path.join(OUT_DIR, "ta_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out_path}")


def plot_stage3_label_by_condition(df_stage3: pd.DataFrame):
    if df_stage3 is None or COL_PART not in df_stage3.columns or COL_LABEL not in df_stage3.columns:
        raise ValueError(f"Stage3 CSV must contain columns: {COL_PART}, {COL_LABEL}")

    # Normalize label names
    df = df_stage3.copy()
    df[COL_LABEL] = df[COL_LABEL].astype(str).str.upper()

    # Count labels by part
    pivot = (
        df.pivot_table(index=COL_PART, columns=COL_LABEL, aggfunc="size", fill_value=0)
          .sort_index()
    )

    # Keep a consistent column order if present
    for col in ["NOT_RANDOM", "RANDOM", "INVALID"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["NOT_RANDOM", "RANDOM", "INVALID"]]

    plt.figure()
    pivot.plot(kind="bar")
    plt.xlabel("Condition (part)")
    plt.ylabel("Number of trials")
    plt.title("Stage 3: Randomness label distribution by condition")
    plt.legend(title="Label")

    out_path = os.path.join(OUT_DIR, "label_distribution_by_condition.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out_path}")


def main():
    df_stage3 = safe_read_csv(STAGE3_CSV)
    if df_stage3 is None:
        raise FileNotFoundError(f"Stage3 CSV not found: {STAGE3_CSV}")

    df_qc = safe_read_csv(STAGE1_TRIAL_QC_CSV)
    df_ta = safe_read_csv(STAGE2_TA_CSV)

    plot_stage1_invalid_distribution(df_qc)
    plot_stage2_ta_distribution(df_ta)
    plot_stage3_label_by_condition(df_stage3)

    print("[DONE] Figures generated.")


if __name__ == "__main__":
    main()