"""
Seed sweep: trains LSTM with multiple seeds and summarises BAcc results.
Usage: python scripts/seed_sweep.py [--model lstm]
"""
import sys, os, subprocess, json, glob, re, time

SEEDS = [0, 7, 13, 21, 37, 77, 100, 200, 300, 512]
MODEL = 'lstm'

for i, arg in enumerate(sys.argv):
    if arg == '--model' and i + 1 < len(sys.argv):
        MODEL = sys.argv[i + 1]

PYTHON = os.path.join(os.path.dirname(__file__), '..', '..\.venv', 'Scripts', 'python.exe')
PIPELINE = os.path.join(os.path.dirname(__file__), '..', 'run_pipeline_simple.py')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'reports')

results = []
print(f"{'Seed':>6} | {'BAcc':>7} | {'Acc':>7} | {'Spec':>7} | {'Rec':>7} | CM")
print("-" * 75)

for seed in SEEDS:
    # Record mtime of existing reports so we can find the newly created one
    existing = set(glob.glob(os.path.join(REPORTS_DIR, f'{MODEL}_*.json')))
    t0 = time.time()
    proc = subprocess.run(
        [PYTHON, PIPELINE, '--model', MODEL, '--seed', str(seed)],
        capture_output=True, text=True,
        cwd=os.path.dirname(PIPELINE)
    )
    elapsed = time.time() - t0

    # Find report created during this run
    after = set(glob.glob(os.path.join(REPORTS_DIR, f'{MODEL}_*.json')))
    new_files = after - existing
    if not new_files:
        # Fallback: pick the most recently modified
        all_files = after
        report_file = max(all_files, key=os.path.getmtime) if all_files else None
    else:
        report_file = list(new_files)[0]

    if report_file is None:
        print(f"{seed:>6} | NO REPORT FOUND")
        continue

    try:
        with open(report_file) as f:
            rep = json.load(f)
        metrics = rep.get('metrics', {}).get('test', {})
        acc  = metrics.get('accuracy', float('nan'))
        rec  = metrics.get('recall', float('nan'))
        cm   = metrics.get('confusion_matrix', [[0,0],[0,0]])
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        bacc = (spec + rec) / 2 if spec == spec and rec == rec else float('nan')
        print(f"{seed:>6} | {bacc*100:>6.2f}% | {acc*100:>6.2f}% | {spec*100:>6.2f}% | {rec*100:>6.2f}% | {cm}  ({elapsed:.0f}s)")
        results.append({'seed': seed, 'bacc': bacc, 'acc': acc, 'cm': cm, 'report': report_file})
    except Exception as e:
        print(f"{seed:>6} | ERROR reading report: {e}")

if results:
    best = max(results, key=lambda r: r['bacc'])
    print(f"\nBest seed: {best['seed']}  BAcc={best['bacc']*100:.2f}%  Acc={best['acc']*100:.2f}%")
    print(f"CM: {best['cm']}")
