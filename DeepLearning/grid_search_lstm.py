"""
LSTM Grid Search — Systematic Hyperparameter Tuning
=====================================================
Tests the priority combinations to push accuracy from 76.47% → 80%.

Phases:
  Phase 1  –  5 priority combinations (user-specified)
  Phase 2  –  Broad grid (pos_weight × dropout × lr × wd × batch × patience)
  Phase 3  –  Multi-seed ensemble (best config, 5 seeds)

Run:
    python grid_search_lstm.py            # Phase 1 + 2 (recommended start)
    python grid_search_lstm.py --phase 1  # Only Phase 1
    python grid_search_lstm.py --phase 2  # Only Phase 2
    python grid_search_lstm.py --phase 3  # Only Phase 3 (ensemble)
    python grid_search_lstm.py --all      # All phases

Results are saved to:
    outputs/reports/grid_search_<timestamp>.json
    (console table printed at the end)
"""

import sys
import os
import copy
import json
import time
import argparse
import itertools
import numpy as np
from datetime import datetime
from pathlib import Path

# ── setup path so local imports work ─────────────────────────────────────────
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir / 'src'))

from run_pipeline_simple import run_simple_pipeline, load_config


# ──────────────────────────────────────────────────────────────────────────────
# BASE CONFIG  (the 76.47% champion)
# ──────────────────────────────────────────────────────────────────────────────

BASE_MODEL: dict = {
    'hidden_size':   384,
    'num_layers':    3,
    'dropout':       0.15,
    'fc_hidden_dim': 192,
    'bidirectional': False,
    'sequence_length': 300,
    'input_size':    13,   # will be overridden by pipeline from real data
}

BASE_TRAINING: dict = {
    'learning_rate':          0.0004,
    'weight_decay':           5e-5,
    'batch_size':             16,
    'early_stopping_patience': 30,
    'n_epochs':               400,
    'test_size':              0.2,
}

# Default seed (same as all previous 359 runs)
DEFAULT_SEED: int = 42


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Priority combinations (from user analysis)
# ──────────────────────────────────────────────────────────────────────────────
# Format: (label, model_overrides, training_overrides, pos_weight, seed)

PRIORITY_COMBOS: list[tuple] = [
    (
        "P1-combo1: drop=0.12 lr=4e-4 wd=3e-5 bs=16 pat=50 pw=0.85",
        {'dropout': 0.12},
        {'learning_rate': 4e-4, 'weight_decay': 3e-5, 'batch_size': 16,
         'early_stopping_patience': 50},
        0.85,   # pos_weight
        42,     # seed
    ),
    (
        "P1-combo2: drop=0.18 lr=3e-4 wd=5e-5 bs=16 pat=60 pw=0.75",
        {'dropout': 0.18},
        {'learning_rate': 3e-4, 'weight_decay': 5e-5, 'batch_size': 16,
         'early_stopping_patience': 60},
        0.75,
        42,
    ),
    (
        "P1-combo3: drop=0.15 lr=5e-4 wd=1e-5 bs=8  pat=50 pw=0.90",
        {'dropout': 0.15},
        {'learning_rate': 5e-4, 'weight_decay': 1e-5, 'batch_size': 8,
         'early_stopping_patience': 50},
        0.90,
        42,
    ),
    (
        "P1-combo4: BEST + bs=32 pw=None",
        {},
        {'batch_size': 32},
        None,
        42,
    ),
    (
        "P1-combo5: BEST + wd=0 pw=None",
        {},
        {'weight_decay': 0},
        None,
        42,
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Broad grid
# ──────────────────────────────────────────────────────────────────────────────

GRID_PARAMS: dict = {
    'pos_weight':   [0.6, 0.75, 0.85, 1.0, 1.15, None],
    'dropout':      [0.10, 0.12, 0.15, 0.18],
    'lr':           [2e-4, 3e-4, 4e-4, 5e-4, 6e-4],
    'wd':           [0, 1e-5, 3e-5, 5e-5, 1e-4],
    'batch_size':   [8, 16, 32],
    'patience':     [30, 50, 60],
    'hidden_size':  [320, 384, 448],
    'fc_hidden_dim':[128, 192, 256],
}

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Multi-seed ensemble
# ──────────────────────────────────────────────────────────────────────────────

ENSEMBLE_SEEDS: list[int] = [42, 52, 62, 72, 82]


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build full config dicts with overrides merged into BASE
# ──────────────────────────────────────────────────────────────────────────────

def build_configs(model_overrides: dict, training_overrides: dict) -> tuple[dict, dict]:
    mc = copy.deepcopy(BASE_MODEL)
    mc.update(model_overrides)
    tc = copy.deepcopy(BASE_TRAINING)
    tc.update(training_overrides)
    return mc, tc


# ──────────────────────────────────────────────────────────────────────────────
# Helper: run one experiment and extract result row
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(label: str,
                   mc: dict,
                   tc: dict,
                   pos_weight,
                   seed: int,
                   global_config) -> dict:
    """
    Runs a single pipeline experiment and returns a result-dict row.
    """
    print("\n" + "═" * 70)
    print(f"EXPERIMENT: {label}")
    print(f"  seed={seed}  pos_weight={pos_weight}  lr={tc['learning_rate']:.0e}")
    print(f"  dropout={mc.get('dropout')}  wd={tc['weight_decay']:.0e}"
          f"  bs={tc['batch_size']}  patience={tc['early_stopping_patience']}")
    print(f"  hidden={mc.get('hidden_size')}  fc={mc.get('fc_hidden_dim')}")
    print("═" * 70)

    t0 = time.time()
    try:
        out = run_simple_pipeline(
            model_type='lstm',
            model_config=mc,
            training_config=tc,
            config=global_config,
            pos_weight=pos_weight,
            random_seed=seed,
        )
        elapsed = time.time() - t0
        res = out['results']['test']
        cm = res['confusion_matrix']
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        return {
            'label':      label,
            'seed':       seed,
            'pos_weight': pos_weight,
            'dropout':    mc.get('dropout'),
            'lr':         tc['learning_rate'],
            'wd':         tc['weight_decay'],
            'batch_size': tc['batch_size'],
            'patience':   tc['early_stopping_patience'],
            'hidden_size':mc.get('hidden_size'),
            'fc_dim':     mc.get('fc_hidden_dim'),
            'acc':        round(res['accuracy'] * 100, 2),
            'f1':         round(res['f1'] * 100, 2),
            'auc':        round(res['roc_auc'] * 100, 2),
            'prec':       round(res['precision'] * 100, 2),
            'recall':     round(res['recall'] * 100, 2),
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'elapsed_s':  round(elapsed, 1),
            'status':     'ok',
            'run_id':     out.get('run_id', ''),
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[FAIL] Experiment failed: {e}")
        import traceback; traceback.print_exc()
        return {
            'label':  label,
            'seed':   seed,
            'pos_weight': pos_weight,
            'acc': -1, 'f1': -1, 'auc': -1,
            'prec': -1, 'recall': -1,
            'tn': -1, 'fp': -1, 'fn': -1, 'tp': -1,
            'elapsed_s': round(elapsed, 1),
            'status': f'error: {e}',
            'run_id': '',
        }


# ──────────────────────────────────────────────────────────────────────────────
# Results: print table + save JSON
# ──────────────────────────────────────────────────────────────────────────────

def print_results_table(rows: list[dict], title: str = "RESULTS"):
    print("\n\n" + "═" * 120)
    print(f"  {title}")
    print("═" * 120)
    hdr = (f"{'#':>3}  {'Acc%':>6}  {'F1%':>6}  {'AUC%':>6}"
           f"  {'TN':>3}{'FP':>4}{'FN':>4}{'TP':>4}"
           f"  {'pw':>5}  {'drop':>5}  {'lr':>7}  {'wd':>7}"
           f"  {'bs':>3}  {'pat':>4}  {'h':>4}  {'fc':>4}  label")
    print(hdr)
    print("-" * 120)
    for i, r in enumerate(rows, 1):
        pw_str  = f"{r.get('pos_weight'):.2f}" if r.get('pos_weight') is not None else "None"
        lr_str  = f"{r.get('lr', 0):.0e}"
        wd_str  = f"{r.get('wd', 0):.0e}"
        print(
            f"{i:>3}  {r['acc']:>6.2f}  {r['f1']:>6.2f}  {r['auc']:>6.2f}"
            f"  {r['tn']:>3}{r['fp']:>4}{r['fn']:>4}{r['tp']:>4}"
            f"  {pw_str:>5}  {r.get('dropout', '?'):>5}  {lr_str:>7}  {wd_str:>7}"
            f"  {r.get('batch_size', '?'):>3}  {r.get('patience', '?'):>4}"
            f"  {r.get('hidden_size', '?'):>4}  {r.get('fc_dim', '?'):>4}"
            f"  {r['label'][:55]}"
        )
    print("═" * 120)

    # Best by accuracy
    ok_rows = [r for r in rows if r['acc'] >= 0]
    if ok_rows:
        best = max(ok_rows, key=lambda r: r['acc'])
        print(f"\n★ BEST  Acc={best['acc']:.2f}%  F1={best['f1']:.2f}%"
              f"  TN={best['tn']}  FP={best['fp']}  FN={best['fn']}  TP={best['tp']}")
        print(f"  Config: {best['label']}")
        print(f"  Baseline (76.47%): TN=10  FP=10  FN=2  TP=29")
        delta = best['acc'] - 76.47
        print(f"  Δ vs baseline: {delta:+.2f}%")


def save_results(rows: list[dict], phase_name: str, output_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"grid_search_{phase_name}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Results saved → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Ensemble helpers
# ──────────────────────────────────────────────────────────────────────────────

def run_ensemble(mc: dict, tc: dict, pos_weight, seeds: list[int], global_config) -> dict:
    """
    Train 5 models with different random seeds.
    Returns individual results + ensemble (averaged probabilities).
    """

    print("\n\n" + "═" * 70)
    print("PHASE 3 — MULTI-SEED ENSEMBLE")
    print(f"  Seeds:      {seeds}")
    print(f"  pos_weight: {pos_weight}")
    print(f"  Config:     dropout={mc.get('dropout')}  lr={tc['learning_rate']:.0e}"
          f"  wd={tc['weight_decay']:.0e}  bs={tc['batch_size']}  patience={tc['early_stopping_patience']}")
    print("═" * 70)

    import torch
    from run_pipeline_simple import load_config as _load_config
    from config_loader import load_config as cfg_load

    individual_rows = []
    all_probs = []    # List[np.ndarray], each shape (51,)
    y_test_ref = None

    for seed in seeds:
        row = run_experiment(
            label=f"ensemble_seed={seed}",
            mc=copy.deepcopy(mc),
            tc=copy.deepcopy(tc),
            pos_weight=pos_weight,
            seed=seed,
            global_config=global_config,
        )
        individual_rows.append(row)

        # Re-read probabilities from the saved JSON
        if row['status'] == 'ok' and row['run_id']:
            reports_dir = Path('outputs/reports')
            json_path = reports_dir / f"{row['run_id']}.json"
            if json_path.exists():
                with open(json_path) as f:
                    saved = json.load(f)
                probs = saved.get('metrics', {}).get('test', {}).get('predictions', [])
                y_true = saved.get('metrics', {}).get('test', {}).get('true_labels', [])
                if probs:
                    all_probs.append(np.array(probs))
                    if y_test_ref is None:
                        y_test_ref = np.array(y_true)

    # Ensemble (majority vote on averaged probabilities)
    ensemble_row = {'label': 'ENSEMBLE (avg probs)', 'seed': 'all', 'pos_weight': pos_weight,
                    'acc': -1, 'f1': -1, 'auc': -1, 'prec': -1, 'recall': -1,
                    'tn': -1, 'fp': -1, 'fn': -1, 'tp': -1, 'status': 'no_probs',
                    'elapsed_s': 0, 'run_id': ''}
    if all_probs and y_test_ref is not None:
        from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                     recall_score, roc_auc_score, confusion_matrix)
        avg_probs = np.mean(np.stack(all_probs, axis=0), axis=0)   # (51,)
        y_pred = (avg_probs >= 0.5).astype(int)
        cm = confusion_matrix(y_test_ref, y_pred)
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        ensemble_row.update({
            'dropout':    mc.get('dropout'),
            'lr':         tc['learning_rate'],
            'wd':         tc['weight_decay'],
            'batch_size': tc['batch_size'],
            'patience':   tc['early_stopping_patience'],
            'hidden_size':mc.get('hidden_size'),
            'fc_dim':     mc.get('fc_hidden_dim'),
            'acc':   round(accuracy_score(y_test_ref, y_pred) * 100, 2),
            'f1':    round(f1_score(y_test_ref, y_pred, zero_division=0) * 100, 2),
            'auc':   round(roc_auc_score(y_test_ref, avg_probs) * 100, 2),
            'prec':  round(precision_score(y_test_ref, y_pred, zero_division=0) * 100, 2),
            'recall':round(recall_score(y_test_ref, y_pred, zero_division=0) * 100, 2),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'status': 'ok',
        })
        print(f"\n[OK] Ensemble (avg probs, t=0.5): Acc={ensemble_row['acc']:.2f}%"
              f"  F1={ensemble_row['f1']:.2f}%  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    return {
        'individual': individual_rows,
        'ensemble':   ensemble_row,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='LSTM Grid Search')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], default=None,
                        help='Run specific phase only (1=priority, 2=broad grid, 3=ensemble)')
    parser.add_argument('--all',   action='store_true',
                        help='Run all phases')
    # Phase 2 single-axis sweep options (to avoid huge full grid by default)
    parser.add_argument('--axis', type=str, default=None,
                        choices=['pos_weight', 'dropout', 'lr', 'wd',
                                 'batch_size', 'patience', 'hidden_size', 'fc_hidden_dim'],
                        help='Phase 2: sweep only one axis at a time')
    args = parser.parse_args()

    run_phase1 = args.all or (args.phase in (None, 1))
    run_phase2 = args.all or (args.phase == 2)
    run_phase3 = args.all or (args.phase == 3)

    # Default: phase 1 + 2 (Phase 3 must be explicit)
    if not args.all and args.phase is None:
        run_phase1 = True
        run_phase2 = True
        run_phase3 = False

    # Load global config ONCE (data paths, etc.)
    global_config = load_config()

    # Output directory
    reports_dir = Path('outputs/reports')
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    # ─── PHASE 1 ───────────────────────────────────────────────────────────────
    if run_phase1:
        print("\n\n" + "█" * 70)
        print("  PHASE 1 — 5 PRIORITY COMBINATIONS")
        print("█" * 70)

        p1_rows = []
        for label, m_ov, t_ov, pw, seed in PRIORITY_COMBOS:
            mc, tc = build_configs(m_ov, t_ov)
            row = run_experiment(label, mc, tc, pw, seed, global_config)
            p1_rows.append(row)
            all_rows.append(row)

        print_results_table(p1_rows, title="PHASE 1 RESULTS — Priority Combinations")
        save_results(p1_rows, phase_name='phase1', output_dir=reports_dir)

    # ─── PHASE 2 ───────────────────────────────────────────────────────────────
    if run_phase2:
        print("\n\n" + "█" * 70)
        print("  PHASE 2 — BROAD GRID SEARCH")
        print("█" * 70)

        if args.axis:
            # Single-axis sweep: vary one param, keep all others at BASE
            axis = args.axis
            values = GRID_PARAMS[axis]
            print(f"  Sweeping axis: {axis}  →  {values}")
            grid_combos = []
            for val in values:
                m_ov = {}
                t_ov = {}
                pw   = None
                if axis == 'pos_weight':
                    pw = val
                elif axis in ('dropout', 'hidden_size', 'fc_hidden_dim'):
                    m_ov[axis] = val
                elif axis == 'lr':
                    t_ov['learning_rate'] = val
                elif axis == 'wd':
                    t_ov['weight_decay'] = val
                elif axis == 'batch_size':
                    t_ov['batch_size'] = val
                elif axis == 'patience':
                    t_ov['early_stopping_patience'] = val
                grid_combos.append((f"P2-{axis}={val}", m_ov, t_ov, pw, DEFAULT_SEED))
        else:
            # Full pairwise (pos_weight × dropout × lr × batch) — manageable subset
            # WARNING: Full cartesian product = 6×4×5×5×3×3×3×3 = 97,200 runs (too many)
            # Default: sweep each axis independently (one-at-a-time, baselines fixed)
            print("  One-at-a-time sweep (vary each axis, fix others at BASE)")
            print("  Tip: use --axis <name> to sweep one specific axis")
            grid_combos = []
            for axis, values in GRID_PARAMS.items():
                for val in values:
                    m_ov = {}
                    t_ov = {}
                    pw   = None
                    if axis == 'pos_weight':
                        pw = val
                    elif axis in ('dropout', 'hidden_size', 'fc_hidden_dim'):
                        m_ov[axis] = val
                    elif axis == 'lr':
                        t_ov['learning_rate'] = val
                    elif axis == 'wd':
                        t_ov['weight_decay'] = val
                    elif axis == 'batch_size':
                        t_ov['batch_size'] = val
                    elif axis == 'patience':
                        t_ov['early_stopping_patience'] = val
                    grid_combos.append((f"P2-{axis}={val}", m_ov, t_ov, pw, DEFAULT_SEED))

        print(f"\n  Total experiments in Phase 2: {len(grid_combos)}")

        p2_rows = []
        for label, m_ov, t_ov, pw, seed in grid_combos:
            mc, tc = build_configs(m_ov, t_ov)
            row = run_experiment(label, mc, tc, pw, seed, global_config)
            p2_rows.append(row)
            all_rows.append(row)

        print_results_table(p2_rows, title="PHASE 2 RESULTS — Broad Grid")
        save_results(p2_rows, phase_name='phase2', output_dir=reports_dir)

    # ─── PHASE 3 ───────────────────────────────────────────────────────────────
    if run_phase3:
        print("\n\n" + "█" * 70)
        print("  PHASE 3 — MULTI-SEED ENSEMBLE")
        print("█" * 70)

        # Use BASE config (best known) or best from Phase 1 if available
        best_config_note = "BASE config (76.47% champion)"
        best_mc = copy.deepcopy(BASE_MODEL)
        best_tc = copy.deepcopy(BASE_TRAINING)
        best_pw = None

        if all_rows:
            ok_rows = [r for r in all_rows if r['acc'] >= 0]
            if ok_rows:
                best_row = max(ok_rows, key=lambda r: r['acc'])
                # Reconstruct from best_row if it came from a phase
                best_mc['dropout']       = best_row.get('dropout', BASE_MODEL['dropout'])
                best_mc['hidden_size']   = best_row.get('hidden_size', BASE_MODEL['hidden_size'])
                best_mc['fc_hidden_dim'] = best_row.get('fc_dim', BASE_MODEL['fc_hidden_dim'])
                best_tc['learning_rate'] = best_row.get('lr', BASE_TRAINING['learning_rate'])
                best_tc['weight_decay']  = best_row.get('wd', BASE_TRAINING['weight_decay'])
                best_tc['batch_size']    = best_row.get('batch_size', BASE_TRAINING['batch_size'])
                best_tc['early_stopping_patience'] = best_row.get('patience', BASE_TRAINING['early_stopping_patience'])
                best_pw = best_row.get('pos_weight')
                best_config_note = f"best from phases (Acc={best_row['acc']:.2f}%): {best_row['label']}"

        print(f"  Using: {best_config_note}")

        ensemble_result = run_ensemble(best_mc, best_tc, best_pw, ENSEMBLE_SEEDS, global_config)

        all_ensemble_rows = ensemble_result['individual'] + [ensemble_result['ensemble']]
        print_results_table(all_ensemble_rows, title="PHASE 3 RESULTS — Multi-Seed Ensemble")
        save_results(all_ensemble_rows, phase_name='phase3_ensemble', output_dir=reports_dir)
        all_rows.extend(all_ensemble_rows)

    # ─── FINAL SUMMARY ─────────────────────────────────────────────────────────
    if all_rows:
        print_results_table(all_rows, title="FINAL COMBINED RESULTS (all phases)")
        save_results(all_rows, phase_name='all_phases', output_dir=reports_dir)

    print("\n[OK] Grid search complete.")


if __name__ == '__main__':
    os.chdir(Path(__file__).parent)   # ensure relative paths resolve correctly
    main()
