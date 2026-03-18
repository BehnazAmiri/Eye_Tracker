"""
Quick analysis of all JSON reports in outputs/reports
Computes metrics from raw predictions/probabilities stored in each file
"""
import json
import glob
import os
import sys
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, balanced_accuracy_score
)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "reports")
TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 20

files = sorted(
    glob.glob(os.path.join(REPORTS_DIR, "*.json")),
    key=os.path.getmtime,
    reverse=True
)[:TOP_N]

results = []
for fpath in files:
    try:
        with open(fpath, encoding="utf-8") as fp:
            d = json.load(fp)
    except Exception as e:
        print(f"ERROR reading {fpath}: {e}")
        continue

    name = os.path.basename(fpath).replace(".json", "")

    if isinstance(d, list):
        print(f"{name:<44}  [list format - skip]")
        continue

    # metrics are nested under d["metrics"]["test"]
    raw_metrics = d.get("metrics", {})
    m = raw_metrics.get("test", raw_metrics)   # prefer test split
    preds = m.get("predicted_labels") or m.get("predictions")
    trues = m.get("true_labels")
    probs = m.get("probabilities") or m.get("probability_scores")

    # Extra info from data_config
    dc = d.get("data_config", {})
    seq_len = dc.get("sequence_length", "?")
    parts = ",".join(dc.get("parts_filter", [])) or "?"
    n_test = dc.get("n_test", "?")

    if preds is None or trues is None:
        print(f"{name:<44}  [no pred array] metrics.test keys={list(m.keys())}")
        continue

    preds = [int(x) for x in preds]
    trues = [int(x) for x in trues]

    acc  = accuracy_score(trues, preds)
    f1v  = f1_score(trues, preds, zero_division=0)
    bacc = balanced_accuracy_score(trues, preds)
    cm   = confusion_matrix(trues, preds)

    if cm.shape == (2, 2):
        tn, fp_, fn, tp = cm.ravel()
        spec = tn / (tn + fp_) if (tn + fp_) > 0 else 0
        rec  = tp / (tp + fn)  if (tp + fn)  > 0 else 0
    else:
        tn = fp_ = fn = tp = spec = rec = 0

    auc = 0.0
    if probs is not None:
        try:
            auc = roc_auc_score(trues, probs)
        except Exception:
            pass
    else:
        try:
            auc = roc_auc_score(trues, preds)
        except Exception:
            pass

    results.append((acc, auc, f1v, bacc, spec, rec, tn, fp_, fn, tp, name, seq_len, n_test))

# Sort by balanced accuracy descending (more fair for imbalanced classes)
results.sort(key=lambda x: x[3], reverse=True)

print(f"\n{'Model/Timestamp':<44} {'Acc':>6} {'AUC':>6} {'F1':>6} {'BAcc':>6} {'Spec':>6} {'Rec':>6} {'seq':>4} {'n':>3}  CM")
print("=" * 130)
for r in results:
    acc, auc, f1v, bacc, spec, rec, tn, fp_, fn, tp, name, seq_len, n_test = r
    cm_str = f"[[{tn},{fp_}],[{fn},{tp}]]"
    print(f"{name:<44} {acc:>6.4f} {auc:>6.4f} {f1v:>6.4f} {bacc:>6.4f} {spec:>6.4f} {rec:>6.4f} {str(seq_len):>4} {str(n_test):>3}  {cm_str}")

print(f"\nTotal reports shown: {len(results)}")

# Best by each metric
if results:
    best_acc  = max(results, key=lambda x: x[0])
    best_auc  = max(results, key=lambda x: x[1])
    best_bacc = max(results, key=lambda x: x[3])
    print(f"\nBest Acc : {best_acc[0]:.4f}  BAcc={best_acc[3]:.4f} AUC={best_acc[1]:.4f} @ {best_acc[10]}")
    print(f"Best AUC : {best_auc[1]:.4f}  Acc={best_auc[0]:.4f}  BAcc={best_auc[3]:.4f} @ {best_auc[10]}")
    print(f"Best BAcc: {best_bacc[3]:.4f}  Acc={best_bacc[0]:.4f} AUC={best_bacc[1]:.4f} @ {best_bacc[10]}")
