"""
Comprehensive HTML + CSV report of ALL Deep Learning experiments.

Run from: d:\MasterThesis\MasterThesis\DeepLearning\
Output:   outputs/reports/experiment_report.html
          outputs/reports/experiment_report.csv
"""
import json, glob, os
import pandas as pd
import numpy as np
from datetime import datetime

REPORTS_DIR = 'outputs/reports'
OUT_HTML = os.path.join(REPORTS_DIR, 'experiment_report.html')
OUT_CSV  = os.path.join(REPORTS_DIR, 'experiment_report.csv')

# ─────────────────────────────────────────────────────────────────────────────
# 1. Read all JSON files
# ─────────────────────────────────────────────────────────────────────────────
rows = []
files = sorted(glob.glob(os.path.join(REPORTS_DIR, '*.json')))

for fpath in files:
    try:
        with open(fpath, encoding='utf-8') as fh:
            content = fh.read().strip()
        if not content:
            continue
        d = json.loads(content)
        if not isinstance(d, dict):
            continue
    except Exception:
        continue

    fname = os.path.basename(fpath)
    model_type = d.get('model_type', fname.split('_')[0]).upper()
    ts_raw = d.get('timestamp', '')
    try:
        ts = datetime.fromisoformat(ts_raw).strftime('%Y-%m-%d %H:%M')
    except Exception:
        ts = ts_raw[:16].replace('T', ' ')

    metrics = d.get('metrics', {})
    test_m  = metrics.get('test', metrics)
    if not isinstance(test_m, dict):
        continue

    acc     = test_m.get('accuracy', None)
    recall  = test_m.get('recall', None)
    f1      = test_m.get('f1', None)
    roc_auc = test_m.get('roc_auc', None)
    cm      = test_m.get('confusion_matrix', None)

    if cm and len(cm) == 2:
        tn, fp_ = cm[0][0], cm[0][1]
        fn, tp  = cm[1][0], cm[1][1]
        spec = tn / (tn + fp_) if (tn + fp_) > 0 else None
        rec2 = tp / (tp + fn)  if (tp + fn)  > 0 else None
        bacc = (spec + rec2) / 2 if (spec is not None and rec2 is not None) else None
        cm_str = f"TN={tn} FP={fp_} FN={fn} TP={tp}"
    else:
        spec = rec2 = bacc = None
        cm_str = ''

    tr = d.get('training_config', {})
    mc = d.get('model_config', {})
    dc = d.get('data_config', {})
    cs = d.get('config_snapshot', {})

    # DM config is stored separately in outputs/dl_inputs/<run_id>_filtered_trials/dm_config_snapshot.json
    run_id = d.get('run_id', fname.replace('.json', ''))
    dm_snapshot_path = os.path.join('outputs', 'dl_inputs', run_id + '_filtered_trials', 'dm_config_snapshot.json')
    if os.path.exists(dm_snapshot_path):
        try:
            dm = json.loads(open(dm_snapshot_path, encoding='utf-8').read())
        except Exception:
            dm = {}
    else:
        dm = d.get('dm_source_config', {}) if isinstance(d.get('dm_source_config'), dict) else {}

    # ── Model-specific HP from config_snapshot fallback ──
    model_key = 'MODEL_' + model_type  # e.g. MODEL_LSTM, MODEL_HYBRID
    cs_model = cs.get(model_key, {}) if isinstance(cs, dict) else {}

    def hp(key, *alt_keys):
        """Read HP: try mc then cs_model for primary key, then mc then cs_model for each alt_key."""
        for k in (key,) + alt_keys:
            v = mc.get(k)
            if v is not None: return v
            v = cs_model.get(k)
            if v is not None: return v
        return None

    # ── Model-aware Hidden / Layers ──────────────────────
    # Each architecture uses different field names for its "size" and "depth"
    if model_type in ('LSTM', 'BILSTM'):
        _hidden = hp('hidden_size')
        _layers = hp('num_layers')
    elif model_type in ('CNN', 'CNN1D'):
        _hidden = hp('base_channels')          # channels = effective "hidden size"
        _layers = hp('num_conv_layers')
    elif model_type == 'TRANSFORMER':
        _hidden = hp('d_model')
        _layers = hp('num_layers')
    elif model_type == 'HYBRID':
        _hidden = hp('lstm_hidden_size', 'hidden_size')
        _layers = hp('lstm_num_layers', 'num_layers')
    elif model_type == 'MLP':
        hd = hp('hidden_dims')
        _hidden = str(hd) if hd is not None else None  # e.g. "[128, 64, 32]"
        _layers = len(hd) if isinstance(hd, list) else None
    else:
        _hidden = hp('hidden_size', 'd_model')
        _layers = hp('num_layers')

    # ── Seed: data_config → config_snapshot.DATA → fallback 42 ──
    seed = dc.get('random_seed', None)
    if seed is None and isinstance(cs, dict):
        data_cs = cs.get('DATA', {})
        if isinstance(data_cs, dict):
            seed = data_cs.get('random_seed', None)
    if seed is None:
        seed = 42  # default seed used in all runs

    # ── DM thresholds ──
    dm_thr = dm.get('thresholds', {})
    dm_out = dm.get('output_summary', {})

    rows.append({
        'File':      fname.replace('.json', ''),
        'Timestamp': ts,
        'Model':     model_type,
        'BAcc':      round(bacc * 100, 2) if bacc is not None else None,
        'Acc':       round(acc  * 100, 2) if acc  is not None else None,
        'Recall':    round((rec2 if rec2 is not None else (recall or 0)) * 100, 2) if (rec2 is not None or recall is not None) else None,
        'Spec':      round(spec * 100, 2) if spec is not None else None,
        'F1':        round(f1, 3)      if f1      is not None else None,
        'AUC':       round(roc_auc, 3) if roc_auc is not None else None,
        'CM':        cm_str,
        'LR':        tr.get('learning_rate'),
        'Batch':     tr.get('batch_size'),
        'Dropout':   hp('dropout'),
        'Hidden':    _hidden,
        'Layers':    _layers,
        'Epochs':    tr.get('n_epochs'),
        'Patience':  tr.get('early_stopping_patience'),
        'Bidirect':  hp('bidirectional'),
        'Seed':      seed,
        'ArchConfig': None,
        'Trials':    dc.get('total_samples_used'),
        'Train':     dc.get('n_train'),
        'Test':      dc.get('n_test'),
        'NR':        dc.get('original_class_0'),
        'RD':        dc.get('original_class_1'),
        # DM thresholds — only present for March 2026 runs
        'DM_Pct':    dm_thr.get('stage3_threshold_percentile'),
        'DM_ta_ms':  dm_thr.get('ta_window_ms'),
        'DM_Cov':    dm_thr.get('ta_answer_coverage_threshold'),
        'DM_TrialExcl': dm_thr.get('stage1_invalid_pct_threshold'),
        'DM_Excl':   dm_thr.get('stage1_participant_exclusion_threshold'),
        'DM_Date':   dm.get('generated_at', '')[:10] or None,
        'DM_Total':  dm_out.get('total_trials'),
        # Validate DM_NR/DM_RD: reject corrupted values (max possible ~500 trials)
        'DM_NR':     dm_out.get('NOT_RANDOM') if dm_out.get('NOT_RANDOM', 0) < 10000 else None,
        'DM_RD':     dm_out.get('RANDOM')     if dm_out.get('RANDOM', 0)     < 10000 else None,
        'PosWeight':  tr.get('pos_weight'),
    })

df = pd.DataFrame(rows)
df = df.sort_values(['Model', 'BAcc'], ascending=[True, False]).reset_index(drop=True)
df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
print(f"CSV saved: {OUT_CSV}  ({len(df)} runs)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def v(val, dec=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '<span style="color:#ccc">-</span>'
    if isinstance(val, float):
        return f'{val:.{dec}f}'
    return str(val)

def badge(model):
    c = {'LSTM':'#2980b9','BILSTM':'#8e44ad','HYBRID':'#27ae60',
         'CNN':'#e67e22','CNN1D':'#c0392b','MLP':'#7f8c8d','TRANSFORMER':'#16a085'}
    col = c.get(model, '#555')
    return f'<span class="badge" style="background:{col}">{model}</span>'

def bacc_td(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '<td style="color:#bbb;text-align:center">-</td>'
    if   val >= 72: bg, fg = '#1e8449', 'white'
    elif val >= 68: bg, fg = '#27ae60', 'white'
    elif val >= 62: bg, fg = '#f39c12', 'white'
    elif val >= 55: bg, fg = '#e67e22', 'white'
    else:           bg, fg = '#e74c3c', 'white'
    return f'<td style="background:{bg};color:{fg};font-weight:700;text-align:center">{val:.2f}%</td>'

def safe_int(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return '-'
        return str(int(float(x)))
    except Exception:
        return str(x)

def safe_float(x, dec=2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return '-'
        return f'{float(x):.{dec}f}'
    except Exception:
        return str(x)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Table 1: Summary per model
# ─────────────────────────────────────────────────────────────────────────────
summary = df.groupby('Model')['BAcc'].agg(
    Runs='count', Best='max',
    Mean=lambda x: round(x.mean(), 2),
    Std=lambda x:  round(x.std(),  2)
).reset_index().sort_values('Best', ascending=False)

t1 = ''
for _, r in summary.iterrows():
    t1 += f'''<tr>
      <td>{badge(r["Model"])}</td>
      <td style="text-align:center">{int(r["Runs"])}</td>
      <td style="text-align:center;font-weight:700;color:#27ae60">{r["Best"]:.2f}%</td>
      <td style="text-align:center">{r["Mean"]:.2f}%</td>
      <td style="text-align:center;color:#888">{r["Std"]:.2f}%</td>
    </tr>'''

# ─────────────────────────────────────────────────────────────────────────────
# 4. Table 2: Best run per model
# ─────────────────────────────────────────────────────────────────────────────
best_idx = df.groupby('Model')['BAcc'].idxmax()
t2 = ''
for _, r in df.loc[best_idx].sort_values('BAcc', ascending=False).iterrows():
    seed_str = 'None' if r['Seed'] is None or (isinstance(r['Seed'], float) and np.isnan(r['Seed'])) else safe_int(r['Seed'])
    t2 += f'''<tr>
      <td>{badge(r["Model"])}</td>
      {bacc_td(r["BAcc"])}
      <td style="text-align:center">{v(r["Acc"])}%</td>
      <td style="text-align:center">{v(r["Recall"])}%</td>
      <td style="text-align:center">{v(r["Spec"])}%</td>
      <td style="text-align:center">{v(r["F1"],3)}</td>
      <td style="text-align:center">{v(r["AUC"],3)}</td>
      <td style="font-size:11px">{r["CM"]}</td>
      <td style="text-align:center">{v(r["LR"])}</td>
      <td style="text-align:center">{safe_int(r["Batch"])}</td>
      <td style="text-align:center">{v(r["Dropout"])}</td>
      <td style="text-align:center">{safe_int(r["Hidden"])}</td>
      <td style="text-align:center">{safe_int(r["Layers"])}</td>
      <td style="text-align:center">{seed_str}</td>
      <td style="text-align:center">{safe_int(r["Trials"])}</td>
      <td style="text-align:center">{safe_int(r["DM_Pct"])}</td>
      <td style="font-size:11px;color:#666">{r["Timestamp"]}</td>
    </tr>'''

# ─────────────────────────────────────────────────────────────────────────────
# 5. Table 3: LSTM sweep
# ─────────────────────────────────────────────────────────────────────────────
lstm_df = df[df['Model'] == 'LSTM'].copy()
t3 = ''
best_lstm_bacc = lstm_df['BAcc'].max()
for _, r in lstm_df.iterrows():
    is_best = r['BAcc'] == best_lstm_bacc
    row_bg  = 'background:#fffde7' if is_best else ''
    seed_str = 'None' if r['Seed'] is None or (isinstance(r['Seed'], float) and np.isnan(r['Seed'])) else safe_int(r['Seed'])
    t3 += f'''<tr style="{row_bg}">
      <td style="font-size:10px;color:#555">{r["File"][-19:]}</td>
      <td style="font-size:11px">{r["Timestamp"]}</td>
      {bacc_td(r["BAcc"])}
      <td style="text-align:center">{v(r["Acc"])}%</td>
      <td style="text-align:center">{v(r["Recall"])}%</td>
      <td style="text-align:center">{v(r["Spec"])}%</td>
      <td style="text-align:center">{v(r["F1"],3)}</td>
      <td style="text-align:center">{v(r["AUC"],3)}</td>
      <td style="font-size:10px">{r["CM"]}</td>
      <td style="text-align:center">{v(r["LR"])}</td>
      <td style="text-align:center">{safe_int(r["Batch"])}</td>
      <td style="text-align:center">{v(r["Dropout"])}</td>
      <td style="text-align:center">{safe_int(r["Hidden"])}</td>
      <td style="text-align:center">{safe_int(r["Layers"])}</td>
      <td style="text-align:center">{seed_str}</td>
      <td style="text-align:center;font-weight:700">{safe_int(r["DM_Pct"])}</td>
      <td style="text-align:center">{safe_int(r["DM_ta_ms"])}</td>
      <td style="text-align:center">{safe_int(r["Trials"])}</td>
    </tr>'''

# ─────────────────────────────────────────────────────────────────────────────
# 6. Table 4: DM threshold impact
# ─────────────────────────────────────────────────────────────────────────────
dm_key_cols = ['DM_Date', 'DM_Pct', 'DM_ta_ms', 'DM_Cov', 'DM_TrialExcl', 'DM_Excl', 'DM_Total', 'DM_NR', 'DM_RD']
lstm_dm = df[df['Model'] == 'LSTM'].copy()
# Fill NaN strings for groupby
for c in dm_key_cols:
    lstm_dm[c] = lstm_dm[c].fillna('').astype(str)

t4 = ''
for key, grp in lstm_dm.groupby(dm_key_cols):
    dm_date, pct, ta_ms, cov, trial_excl, excl, dm_total, dm_nr, dm_rd = key
    best_b  = grp['BAcc'].max()
    cnt     = len(grp)
    best_f  = grp.loc[grp['BAcc'].idxmax(), 'File'][-19:]
    col     = '#1e8449' if best_b >= 72 else ('#27ae60' if best_b >= 68 else ('#f39c12' if best_b >= 62 else '#e74c3c'))
    t4 += f'''<tr>
      <td style="text-align:center">{dm_date or "-"}</td>
      <td style="text-align:center;font-weight:700">{pct or "-"}</td>
      <td style="text-align:center">{ta_ms or "-"}</td>
      <td style="text-align:center">{cov or "-"}</td>
      <td style="text-align:center">{trial_excl or "-"}</td>
      <td style="text-align:center">{excl or "-"}</td>
      <td style="text-align:center">{dm_total or "-"}</td>
      <td style="text-align:center">{dm_nr or "-"}</td>
      <td style="text-align:center">{dm_rd or "-"}</td>
      <td style="text-align:center">{cnt}</td>
      <td style="text-align:center;background:{col};color:white;font-weight:700">{best_b:.2f}%</td>
      <td style="font-size:10px;color:#666">{best_f}</td>
    </tr>'''

# ─────────────────────────────────────────────────────────────────────────────
# 7. Table 5: All runs
# ─────────────────────────────────────────────────────────────────────────────
t5 = ''
prev_model = None
model_best = df.groupby('Model')['BAcc'].max()
for _, r in df.iterrows():
    if r['Model'] != prev_model:
        n_model_runs = int(df[df['Model'] == r['Model']].shape[0])
        t5 += f'''<tr style="background:#2c3e50">
          <td colspan="18" style="color:white;font-weight:700;padding:8px 14px">
            {badge(r["Model"])} &nbsp; {r["Model"]} &mdash; {n_model_runs} runs &nbsp;
            (Best BAcc: {model_best[r["Model"]]:.2f}%)
          </td></tr>'''
        prev_model = r['Model']
    is_best = r['BAcc'] == model_best[r['Model']]
    row_bg  = 'background:#fffde7' if is_best else ''
    seed_str = 'None' if r['Seed'] is None or (isinstance(r['Seed'], float) and np.isnan(r['Seed'])) else safe_int(r['Seed'])
    t5 += f'''<tr style="{row_bg}">
      <td style="font-size:10px;color:#555">{r["File"][-19:]}</td>
      <td style="font-size:11px">{r["Timestamp"]}</td>
      <td style="text-align:center">{badge(r["Model"])}</td>
      {bacc_td(r["BAcc"])}
      <td style="text-align:center">{v(r["Acc"])}%</td>
      <td style="text-align:center">{v(r["Recall"])}%</td>
      <td style="text-align:center">{v(r["Spec"])}%</td>
      <td style="text-align:center">{v(r["F1"],3)}</td>
      <td style="text-align:center">{v(r["AUC"],3)}</td>
      <td style="font-size:10px">{r["CM"]}</td>
      <td style="text-align:center">{v(r["LR"])}</td>
      <td style="text-align:center">{safe_int(r["Batch"])}</td>
      <td style="text-align:center">{v(r["Dropout"])}</td>
      <td style="text-align:center">{safe_int(r["Hidden"])}</td>
      <td style="text-align:center">{safe_int(r["Layers"])}</td>
      <td style="text-align:center">{seed_str}</td>
      <td style="text-align:center;font-weight:700">{safe_int(r["DM_Pct"])}</td>
      <td style="text-align:center">{safe_int(r["Trials"])}</td>
    </tr>'''

# ─────────────────────────────────────────────────────────────────────────────
# 8. Assemble final HTML
# ─────────────────────────────────────────────────────────────────────────────
total_runs = len(df)
best_bacc  = df['BAcc'].max()
best_model = df.loc[df['BAcc'].idxmax(), 'Model']
best_run   = df.loc[df['BAcc'].idxmax(), 'File']
n_models   = df['Model'].nunique()
models_str = ', '.join(sorted(df['Model'].unique()))
max_trials = int(df['Trials'].dropna().max()) if not df['Trials'].dropna().empty else '-'

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DL Experiment Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',Arial,sans-serif;background:#eef2f7;color:#2c3e50;font-size:13px}}
  .header{{background:linear-gradient(135deg,#1a1a2e,#0f3460);color:white;padding:28px 40px}}
  .header h1{{font-size:22px;font-weight:700;margin-bottom:8px}}
  .header p{{font-size:12px;opacity:.8;line-height:1.8}}
  .wrap{{padding:24px 36px;max-width:1700px;margin:0 auto}}
  .stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}}
  .stat{{background:white;border-radius:8px;padding:18px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,.08)}}
  .stat-n{{font-size:28px;font-weight:700;color:#2c3e50}}
  .stat-l{{font-size:11px;color:#888;margin-top:4px}}
  .card{{background:white;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,.08);margin-bottom:24px;overflow:hidden}}
  .card-h{{background:#34495e;color:white;padding:11px 18px;font-size:13px;font-weight:700}}
  .card-h small{{font-weight:400;opacity:.75;margin-left:10px;font-size:11px}}
  .scroll{{overflow-x:auto}}
  .scroll-y{{overflow-x:auto;max-height:520px;overflow-y:auto}}
  table{{width:100%;border-collapse:collapse;white-space:nowrap}}
  th{{background:#2c3e50;color:white;padding:7px 10px;font-size:11px;font-weight:600;
      position:sticky;top:0;z-index:1;border-right:1px solid #3d5166}}
  td{{padding:5px 10px;border-bottom:1px solid #ecf0f1}}
  tr:hover td{{background:#eaf5fb!important}}
  .badge{{color:white;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;display:inline-block}}
  .legend{{display:flex;gap:14px;padding:10px 18px;background:#f8f9fa;border-top:1px solid #ecf0f1;flex-wrap:wrap}}
  .leg{{display:flex;align-items:center;gap:5px;font-size:11px}}
  .leg-box{{width:13px;height:13px;border-radius:3px}}
  .note{{font-size:11px;color:#666;padding:8px 18px;background:#fffbf0;border-top:1px solid #f0e6c8;line-height:1.6}}
</style>
</head>
<body>
<div class="header">
  <h1>Eye-Tracking Randomness Detection &mdash; Deep Learning Experiment Report</h1>
  <p>
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
    Total experiments: <strong>{total_runs}</strong> &nbsp;|&nbsp;
    Architectures tested: {n_models} &mdash; {models_str}<br>
    Best Balanced Accuracy: <strong>{best_bacc:.2f}%</strong> ({best_model}) &nbsp;|&nbsp;
    Best run: {best_run}
  </p>
</div>

<div class="wrap">

<div class="stats">
  <div class="stat"><div class="stat-n">{total_runs}</div><div class="stat-l">Total Experiments</div></div>
  <div class="stat"><div class="stat-n" style="color:#27ae60">{best_bacc:.1f}%</div><div class="stat-l">Best Balanced Accuracy (BAcc)</div></div>
  <div class="stat"><div class="stat-n" style="color:#2980b9">{n_models}</div><div class="stat-l">Model Architectures Tested</div></div>
  <div class="stat"><div class="stat-n" style="color:#8e44ad">{max_trials}</div><div class="stat-l">Maximum Trials in Dataset</div></div>
</div>

<div class="card">
  <div class="card-h">Table 1 &mdash; Summary by Model Architecture
    <small>BAcc = Balanced Accuracy = (Specificity + Recall) / 2</small>
  </div>
  <table>
    <thead><tr><th>Model</th><th># Runs</th><th>Best BAcc (%)</th><th>Mean BAcc (%)</th><th>Std BAcc (%)</th></tr></thead>
    <tbody>{t1}</tbody>
  </table>
</div>

<div class="card">
  <div class="card-h">Table 2 &mdash; Best Configuration per Model Architecture</div>
  <div class="scroll"><table>
    <thead><tr>
      <th>Model</th><th>BAcc (%)</th><th>Acc (%)</th><th>Recall (%)</th><th>Spec (%)</th>
      <th>F1</th><th>AUC</th><th>Confusion Matrix</th>
      <th>Learning Rate</th><th>Batch</th><th>Dropout</th><th>Hidden</th><th>Layers</th>
      <th>Seed</th><th>Trials</th><th>DM Pct.</th><th>Timestamp</th>
    </tr></thead>
    <tbody>{t2}</tbody>
  </table></div>
  <div class="legend">
    <div class="leg"><div class="leg-box" style="background:#1e8449"></div>BAcc &ge; 72%</div>
    <div class="leg"><div class="leg-box" style="background:#27ae60"></div>BAcc 68&ndash;72%</div>
    <div class="leg"><div class="leg-box" style="background:#f39c12"></div>BAcc 62&ndash;68%</div>
    <div class="leg"><div class="leg-box" style="background:#e67e22"></div>BAcc 55&ndash;62%</div>
    <div class="leg"><div class="leg-box" style="background:#e74c3c"></div>BAcc &lt; 55%</div>
  </div>
</div>

<div class="card">
  <div class="card-h">Table 3 &mdash; LSTM Hyperparameter Sweep &mdash; All {len(lstm_df)} LSTM Runs
    <small>Yellow row = best LSTM result | Sorted by BAcc descending</small>
  </div>
  <div class="scroll-y"><table>
    <thead><tr>
      <th>Run ID</th><th>Timestamp</th><th>BAcc (%)</th><th>Acc (%)</th>
      <th>Recall (%)</th><th>Spec (%)</th><th>F1</th><th>AUC</th><th>Confusion Matrix</th>
      <th>LR</th><th>Batch</th><th>Dropout</th><th>Hidden</th><th>Layers</th>
      <th>Seed</th><th>DM Pct.</th><th>DM ta_ms</th><th>Trials</th>
    </tr></thead>
    <tbody>{t3}</tbody>
  </table></div>
</div>

<div class="card">
  <div class="card-h">Table 4 &mdash; DataMining Threshold Configurations vs. Best LSTM Performance
    <small>Each row = unique DataMining configuration used for labeling | Shows how thresholds affect dataset and best achievable BAcc</small>
  </div>
  <div class="scroll"><table>
    <thead><tr>
      <th>DM Generated</th><th>Pct. Threshold</th><th>ta Window (ms)</th>
      <th>Coverage Thr.</th><th>Excl. Thr.</th>
      <th>DM Total Trials</th><th>DM NOT_RANDOM</th><th>DM RANDOM</th>
      <th>LSTM Runs</th><th>Best BAcc (%)</th><th>Best Run ID</th>
    </tr></thead>
    <tbody>{t4}</tbody>
  </table></div>
  <div class="note">
    <strong>Pct. Threshold</strong> = stage3_threshold_percentile: which percentile of t_answer is used as RANDOM/NOT_RANDOM boundary for Timer-Correct trials (e.g. P25 = 25th percentile &asymp; 25.2s)<br>
    <strong>ta Window (ms)</strong> = ta_window_ms: time window for detecting first fixation on Answer_Area<br>
    <strong>Coverage Thr.</strong> = ta_answer_coverage_threshold: minimum fraction of fixation window that must fall on Answer_Area<br>
    <strong>Excl. Thr.</strong> = stage1_participant_exclusion_threshold: max fraction of invalid trials before participant is excluded from analysis
  </div>
</div>

<div class="card">
  <div class="card-h">Table 5 &mdash; All {total_runs} Experiment Runs (Complete Log)
    <small>Yellow = best per model | Grouped by model type | Sorted by BAcc desc within each group</small>
  </div>
  <div class="scroll-y"><table>
    <thead><tr>
      <th>Run ID</th><th>Timestamp</th><th>Model</th><th>BAcc (%)</th><th>Acc (%)</th>
      <th>Recall (%)</th><th>Spec (%)</th><th>F1</th><th>AUC</th><th>Confusion Matrix</th>
      <th>LR</th><th>Batch</th><th>Dropout</th><th>Hidden</th><th>Layers</th>
      <th>Seed</th><th>DM Pct.</th><th>Trials</th>
    </tr></thead>
    <tbody>{t5}</tbody>
  </table></div>
</div>

</div>
</body>
</html>"""

with open(OUT_HTML, 'w', encoding='utf-8') as fh:
    fh.write(HTML)

print(f"\nHTML: {OUT_HTML}")
print(f"CSV:  {OUT_CSV}")
print(f"\n{'='*60}")
print("TABLE 1 — Summary by Model (sorted by Best BAcc):")
print(summary.to_string(index=False))

print(f"\nTABLE 2 — Best Run per Model:")
cols_t2 = ['Model', 'BAcc', 'Acc', 'Recall', 'Spec', 'F1', 'AUC',
           'LR', 'Batch', 'Dropout', 'Hidden', 'Layers', 'Seed', 'Trials', 'DM_Pct', 'Timestamp']
print(df.loc[best_idx, cols_t2].sort_values('BAcc', ascending=False).to_string(index=False))

print(f"\nTABLE 4 — DM Threshold Impact (LSTM only):")
dm_summary = (df[df['Model'] == 'LSTM']
              .groupby(['DM_Date', 'DM_Pct', 'DM_ta_ms', 'DM_Total', 'DM_NR', 'DM_RD'], dropna=False)['BAcc']
              .agg(Runs='count', BestBAcc='max')
              .reset_index()
              .sort_values('DM_Date'))
print(dm_summary.to_string(index=False))
