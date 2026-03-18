"""
Simple two-table experiment report.
Run from: d:\MasterThesis\MasterThesis\DeepLearning\
Output:   outputs/reports/results_report.html

Table 1 — Best run per model: shows best BAcc row AND best Acc row (if different)
Table 2 — Running log: starts empty, only accumulates runs added AFTER this script was first created.
           Every time you run a new pipeline then rerun this script, the new run appears at the top.
"""
import pandas as pd, re, pathlib, json

# ── Answer Correctness (from DataMining outputs) ──────────────────────────────
# Columns added to both tables: correct-answer rate per Part×Label group
COR_COLS  = ['Cor_NT_RD', 'Cor_NT_NR', 'Cor_TC_RD', 'Cor_TC_NR']
COR_HEADS = ['NT RAND Cor%', 'NT NR Cor%', 'TC RAND Cor%', 'TC NR Cor%']
COR_MAP   = {
    ('No-Timer',       'RANDOM'):     'Cor_NT_RD',
    ('No-Timer',       'NOT_RANDOM'): 'Cor_NT_NR',
    ('Timer-Correct',  'RANDOM'):     'Cor_TC_RD',
    ('Timer-Correct',  'NOT_RANDOM'): 'Cor_TC_NR',
}

def compute_correctness_flat():
    """Returns flat dict: {Cor_NT_RD: (pct, correct, total), ...}"""
    rows_ans = []
    for ans_file in sorted(pathlib.Path('../Eye_tracking_app/outputs').glob('participant_*/answers.json')):
        try:
            answers = json.loads(ans_file.read_text(encoding='utf-8'))
            if isinstance(answers, dict): answers = [answers]
            for a in answers:
                qid = a.get('question_id') or a.get('question_number')
                chosen = a.get('chosen_option', '')
                rows_ans.append({
                    'participant_id': ans_file.parent.name,
                    'question_id': 'question_' + str(qid),
                    'is_correct': int(str(chosen).endswith('-C')) if chosen else 0
                })
        except Exception:
            continue

    stage3_path = pathlib.Path('../DataMining/results/reports/stage3/stage3_with_labels.csv')
    if not rows_ans or not stage3_path.exists():
        return {}

    ans_df = pd.DataFrame(rows_ans)
    s3 = pd.read_csv(stage3_path)
    merged = (s3.drop_duplicates(['participant_id', 'question_id'])
                .merge(ans_df, on=['participant_id', 'question_id'], how='left'))
    merged = merged[merged['randomness_label'].isin(['RANDOM', 'NOT_RANDOM'])]

    flat = {}
    for (part, label), col in COR_MAP.items():
        sub = merged[(merged['part'] == part) & (merged['randomness_label'] == label)]
        if len(sub) == 0: continue
        total   = len(sub)
        correct = int(sub['is_correct'].sum())
        flat[col] = (round(correct / total * 100, 1), correct, total)
    return flat

corr_flat = compute_correctness_flat()

# ── Load all results ──────────────────────────────────────────────────────────
df = pd.read_csv('outputs/reports/experiment_report.csv')
df = df.sort_values('Timestamp', ascending=False).reset_index(drop=True)

# ── Table 1: top 10 by Accuracy — unique Acc values only ────────────────────
combined = (df.sort_values('Acc', ascending=False)
              .drop_duplicates('File')
              .drop_duplicates(['Acc', 'CM'])   # remove only if both Acc AND CM are identical
              .head(10)
              .reset_index(drop=True))

# Add correctness columns (same DM config for all Table 1 rows)
for c in COR_COLS:
    val = corr_flat.get(c)
    combined[c]              = val[0] if val else None
    combined[c + '_correct'] = val[1] if val else None
    combined[c + '_total']   = val[2] if val else None

table1 = combined

# ── Table 2: running log — only new runs (not in seen_files.txt) ─────────────
log_path  = pathlib.Path('outputs/reports/run_log.csv')
seen_path = pathlib.Path('outputs/reports/seen_files.txt')

# Load "seen" baseline (all runs that existed when the log was initialized)
seen_files = set()
if seen_path.exists():
    seen_files = set(seen_path.read_text(encoding='utf-8').splitlines())

# Load existing log
if log_path.exists() and log_path.stat().st_size > 0:
    log_df = pd.read_csv(log_path)
    seen_files.update(log_df['File'].astype(str).tolist())
else:
    log_df = pd.DataFrame()

# Find genuinely new runs
new_runs = df[~df['File'].astype(str).isin(seen_files)].copy()

# Stamp correctness values at time of run (reflects current DM threshold)
for c in COR_COLS:
    val = corr_flat.get(c)
    new_runs[c]              = val[0] if val else None
    new_runs[c + '_correct'] = val[1] if val else None
    new_runs[c + '_total']   = val[2] if val else None

# Append new runs to log
if not new_runs.empty:
    log_df = pd.concat([new_runs, log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)
    print(f'  + {len(new_runs)} new run(s) appended to log.')
else:
    print('  No new runs since last time.')

# Table 2 is whatever is in the log (newest first)
table2 = log_df.sort_values('Timestamp', ascending=False).reset_index(drop=True) if not log_df.empty else pd.DataFrame()

# ── Helpers ───────────────────────────────────────────────────────────────────
def v(val, decimals=2):
    try:
        if val != val: return '—'
        return f'{float(val):.{decimals}f}'
    except:
        return str(val) if val else '—'

def parse_cm(s):
    if not isinstance(s, str): return '—'
    m = re.findall(r'\d+', s)
    if len(m) == 4:
        return f'TN={m[0]} FP={m[1]}<br>FN={m[2]} TP={m[3]}'
    return '—'

def safe_int(val):
    try:
        if val != val: return '—'
        return str(int(float(val)))
    except:
        return '—'

MODEL_COLORS = {
    'LSTM':        '#3b5bdb',
    'HYBRID':      '#0ca678',
    'CNN':         '#e67700',
    'CNN1D':       '#c2255c',
    'TRANSFORMER': '#6741d9',
    'BILSTM':      '#1971c2',
    'MLP':         '#5c7c33',
}

TABLE_ID_COUNTER = [0]

def build_table(rows_df):
    if rows_df is None or rows_df.empty:
        return '<p style="color:#999;padding:12px">No data yet. Run a new pipeline and rerun this script.</p>'

    TABLE_ID_COUNTER[0] += 1
    tid = f'tbl{TABLE_ID_COUNTER[0]}'

    html = f'<table id="{tid}">\n<thead>'
    headers = ['#', 'Model']
    headers += ['Acc', 'Rec', 'Spec', 'F1', 'AUC',
                'LR', 'Batch', 'Drop', 'Hidden', 'Layers', 'Seed', 'PosW',
                'Pct', 'ta ms', 'NR', 'RD', 'Cov', 'TrExcl', 'PtExcl']
    headers += ['NT-RD%', 'NT-NR%', 'TC-RD%', 'TC-NR%']
    headers += ['CM']
    # Sort + filter row
    html += '<tr class="sort-row">'
    for ci, h in enumerate(headers):
        sortable = h not in ('#', 'Confusion Matrix')
        if sortable:
            html += (f'<th>'
                     f'<span class="sort-label" onclick="sortTable(\'{tid}\',{ci})">{h} <span class="sort-icon">⇅</span></span>'
                     f'<button class="filter-btn" onclick="openFilter(event,\'{tid}\',{ci})" title="Filter column">&#9660;</button>'
                     f'</th>')
        else:
            html += f'<th>{h}</th>'
    html += '</tr>\n'
    html += '</thead>\n<tbody>\n'

    for i, (_, row) in enumerate(rows_df.iterrows()):
        color = MODEL_COLORS.get(row.get('Model', ''), '#555')

        acc   = float(row['Acc']) if row['Acc'] == row['Acc'] else 0
        acc_color = '#1a7a1a' if acc >= 74 else ('#d07000' if acc >= 68 else '#c0392b')

        def dm(col):
            val = row.get(col)
            if val is None or val != val: return '<span class="na">—</span>'
            try: return safe_int(val)
            except: return str(val)

        hidden = str(row.get('Hidden', '—')).replace('[','').replace(']','') if row.get('Hidden') == row.get('Hidden') else '—'

        html += f'<tr>'
        html += f'<td class="num">{i+1}</td>'
        html += f'<td><span class="badge" style="color:{color}">{row.get("Model","")}</span></td>'
        html += f'<td class="num" style="color:{acc_color};font-weight:700">{v(row["Acc"])}%</td>'
        html += f'<td class="num" style="color:#1a6e1a">{v(row.get("Recall",""))}%</td>'
        html += f'<td class="num" style="color:#1a5fa8">{v(row.get("Spec",""))}%</td>'
        html += f'<td class="num">{v(row.get("F1",""), 3)}</td>'
        html += f'<td class="num">{v(row.get("AUC",""), 3)}</td>'
        html += f'<td class="num">{row.get("LR","—")}</td>'
        html += f'<td class="num">{safe_int(row.get("Batch",""))}</td>'
        html += f'<td class="num">{row.get("Dropout","—")}</td>'
        html += f'<td class="num">{hidden}</td>'
        html += f'<td class="num">{safe_int(row.get("Layers",""))}</td>'
        html += f'<td class="num">{safe_int(row.get("Seed",""))}</td>'
        pw = row.get('PosWeight')
        pw_str = f'{float(pw):.2f}' if (pw is not None and pw == pw) else '<span class="na">auto</span>'
        html += f'<td class="num" style="color:#5b1fa8">{pw_str}</td>'
        def dmv(col, suffix=''):
            val = row.get(col)
            if val is None or val != val: return '<span class="na">—</span>'
            try:
                f = float(val)
                return f'{f:.2f}{suffix}' if f != int(f) else f'{int(f)}{suffix}'
            except: return str(val)

        html += f'<td class="num dm"><strong>{dmv("DM_Pct", "th")}</strong></td>'
        html += f'<td class="num dm">{dmv("DM_ta_ms", " ms")}</td>'
        html += f'<td class="num dm">{dmv("DM_NR")}</td>'
        html += f'<td class="num dm">{dmv("DM_RD")}</td>'
        html += f'<td class="num dm">{dmv("DM_Cov")}</td>'
        html += f'<td class="num dm">{dmv("DM_TrialExcl")}</td>'
        html += f'<td class="num dm">{dmv("DM_Excl")}</td>'
        for c in COR_COLS:
            val = row.get(c)
            if val is None or val != val:
                html += '<td class="num cor"><span class="na">—</span></td>'
            else:
                pct     = float(val)
                correct = row.get(c + '_correct')
                total   = row.get(c + '_total')
                clr = '#1a7a1a' if pct >= 70 else ('#d07000' if pct >= 55 else '#c0392b')
                count_str = f'<br><span style="font-size:.75em;color:#777">{int(correct)}/{int(total)}</span>' if correct == correct and total == total and correct is not None else ''
                html += f'<td class="num cor" style="color:{clr}">{pct}%{count_str}</td>'
        html += f'<td class="cm">{parse_cm(row.get("CM",""))}</td>'
        html += '</tr>\n'

    html += '</tbody></table>'
    return html

t1 = build_table(table1)
t2 = build_table(table2)

LEGEND = """
<div class="legend">
  <b>Rec</b>=Recall &nbsp;<b>Spec</b>=Specificity &nbsp;<b>Drop</b>=Dropout &nbsp;<b>PosW</b>=pos_weight &nbsp;
  <b style="color:#5b1fa8">Pct</b>=<span style="color:#5b1fa8">Percentile(RANDOM threshold)</span> &nbsp;
  <b style="color:#5b1fa8">NR</b>=<span style="color:#5b1fa8">NOT_RANDOM</span> &nbsp;
  <b style="color:#5b1fa8">RD</b>=<span style="color:#5b1fa8">RANDOM</span> &nbsp;
  <b style="color:#5b1fa8">Cov</b>=<span style="color:#5b1fa8">answer coverage threshold</span> &nbsp;
  <b style="color:#5b1fa8">TrExcl/PtExcl</b>=<span style="color:#5b1fa8">excluded trials/participants</span> &nbsp;
  <b>NT-RD%</b>=No-Timer RANDOM correct% &nbsp;<b>TC-NR%</b>=Timer-Correct NOT_RANDOM correct% &nbsp;
  <b>CM</b>: TN·FP / FN·TP
</div>
"""

log_note = f'{len(table2)} run(s) logged' if not table2.empty else 'empty — waiting for new runs'

# ── (correctness block removed — now shown as columns in both tables) ────────
def build_correctness_block(rows):  # kept for compatibility, not used
    if not rows:
        return '<p style="color:#aaa;font-size:.85em">No data available.</p>'

    parts = ['No-Timer', 'Timer-Correct', 'Timer-No-Correct']
    part_labels = {p: {} for p in parts}
    for r in rows:
        part_labels[r['part']][r['label']] = r

    html  = '<table class="corr-table">'
    html += '<thead><tr>'
    for h in ['Part', 'Label', 'Total Trials', 'Correct', 'Incorrect', 'Correct %', 'Incorrect %']:
        html += f'<th>{h}</th>'
    html += '</tr></thead><tbody>'

    ROW_COLORS = {'No-Timer': '#eef2ff', 'Timer-Correct': '#efffef', 'Timer-No-Correct': '#fff5f5'}

    for part in parts:
        data  = part_labels.get(part, {})
        bg    = ROW_COLORS.get(part, '#fff')
        first = True
        labels_present = [lbl for lbl in ['RANDOM', 'NOT_RANDOM'] if lbl in data]
        rowspan = len(labels_present) + 1   # +1 for TOTAL row

        # compute part totals
        total_t   = sum(data[l]['total']   for l in labels_present)
        total_c   = sum(data[l]['correct'] for l in labels_present)
        total_i   = total_t - total_c
        total_cp  = round(total_c / total_t * 100, 1) if total_t else 0
        total_ip  = round(100 - total_cp, 1)

        for lbl in labels_present:
            r = data[lbl]
            incorrect = r['total'] - r['correct']
            ip = round(100 - r['pct'], 1)
            lbl_style = 'color:#5b1fa8;font-weight:600' if lbl == 'RANDOM' else 'color:#1a5fa8;font-weight:600'
            html += f'<tr style="background:{bg}">'
            if first:
                html += f'<td rowspan="{rowspan}" style="font-weight:700;vertical-align:middle;border-right:2px solid #ddd">{part}</td>'
                first = False
            html += f'<td style="{lbl_style}">{lbl}</td>'
            html += f'<td class="num">{r["total"]}</td>'
            html += f'<td class="num" style="color:#1a6e1a">{r["correct"]}</td>'
            html += f'<td class="num" style="color:#c0392b">{incorrect}</td>'
            html += f'<td class="num" style="color:#1a6e1a;font-weight:600">{r["pct"]}%</td>'
            html += f'<td class="num" style="color:#c0392b">{ip}%</td>'
            html += '</tr>'

        # TOTAL row for part
        html += f'<tr style="background:{bg};border-top:1px solid #ccc;font-weight:700">'
        html += f'<td style="color:#555">TOTAL</td>'
        html += f'<td class="num">{total_t}</td>'
        html += f'<td class="num" style="color:#1a6e1a">{total_c}</td>'
        html += f'<td class="num" style="color:#c0392b">{total_i}</td>'
        html += f'<td class="num" style="color:#1a6e1a">{total_cp}%</td>'
        html += f'<td class="num" style="color:#c0392b">{total_ip}%</td>'
        html += '</tr>'

    html += '</tbody></table>'
    return html

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment Results</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: Segoe UI, Arial, sans-serif; background: #f4f5f7; padding: 16px; color: #222; font-size: 12px; }}
h2 {{ font-size: 1.05em; color: #2c2c5e; margin: 0 0 8px;
     border-left: 4px solid #3b5bdb; padding-left: 10px; }}
.section {{ background: #fff; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,.08);
            padding: 14px 16px; margin-bottom: 20px; overflow-x: auto; }}
.subtitle {{ font-size: .8em; color: #999; margin: -4px 0 10px 14px; }}
table {{ border-collapse: collapse; width: max-content; min-width: 100%; font-size: .8em; }}
thead tr {{ background: #2c2c5e; color: #fff; }}
th {{ padding: 5px 6px; text-align: center; font-weight: 600; white-space: nowrap; }}
tbody tr {{ border-bottom: 1px solid #eee; }}
tbody tr:hover {{ background: #f7f8ff; }}
td {{ padding: 4px 6px; vertical-align: middle; white-space: nowrap; text-align: center; }}
.num  {{ text-align: center; font-variant-numeric: tabular-nums; }}
.cm   {{ font-family: monospace; font-size: .76em; color: #555; max-width: 120px; white-space: normal; word-break: break-word; }}
.ts   {{ font-size: .76em; color: #aaa; }}
.dm   {{ color: #5b1fa8; }}
.na   {{ color: #ddd; }}
.badge {{ font-weight: 700; font-size: .82em; }}
.champ {{ font-size: .76em; font-weight: 600; }}
.cor   {{ font-size: .8em; font-weight: 600; border-left: 1px solid #e8e0f5; }}
.legend {{ font-size: .76em; color: #555; line-height: 1.8; background:#fff; border-radius:8px; padding: 8px 14px; margin-bottom: 14px; box-shadow: 0 1px 6px rgba(0,0,0,.07); }}
.legend span {{ margin-right: 5px; }}
.lg-group {{ font-weight: 700; color: #2c2c5e; }}
.sort-row th {{ user-select: none; white-space: nowrap; }}
.sort-row th:hover {{ background: #3d3d7a; }}
.sort-label {{ cursor: pointer; }}
.sort-icon {{ font-size: .75em; opacity: .6; }}
.filter-btn {{ background: none; border: none; color: #aac; font-size: .7em; cursor: pointer;
               padding: 0 0 0 4px; vertical-align: middle; opacity: .7; }}
.filter-btn:hover {{ opacity: 1; color: #fff; }}
.filter-btn.active {{ color: #7df; opacity: 1; }}
.filtered-out {{ display: none; }}
/* Dropdown */
#flt-drop {{ position: fixed; z-index: 9999; background: #fff; border: 1px solid #ccd; border-radius: 6px;
             box-shadow: 0 4px 18px rgba(0,0,0,.18); min-width: 180px; max-width: 260px;
             font-size: 13px; display: none; }}
#flt-drop .flt-head {{ background: #2c2c5e; color: #fff; padding: 7px 10px; border-radius: 5px 5px 0 0;
                       font-weight: 600; font-size: .85em; }}
#flt-drop .flt-search {{ padding: 6px 8px; border-bottom: 1px solid #eee; }}
#flt-drop .flt-search input {{ width: 100%; border: 1px solid #ccd; border-radius: 4px;
                                padding: 3px 6px; font-size: .85em; outline: none; }}
#flt-drop .flt-items {{ max-height: 220px; overflow-y: auto; padding: 4px 0; }}
#flt-drop .flt-items label {{ display: flex; align-items: center; gap: 7px;
                               padding: 4px 12px; cursor: pointer; }}
#flt-drop .flt-items label:hover {{ background: #f0f0ff; }}
#flt-drop .flt-items input[type=checkbox] {{ cursor: pointer; }}
#flt-drop .flt-actions {{ display: flex; gap: 6px; padding: 7px 10px; border-top: 1px solid #eee; }}
#flt-drop .flt-actions button {{ flex:1; padding: 4px; border: 1px solid #ccd; border-radius: 4px;
                                  background: #f5f5ff; cursor: pointer; font-size: .82em; }}
#flt-drop .flt-actions button:hover {{ background: #2c2c5e; color: #fff; }}
#flt-drop .flt-apply {{ background: #2c2c5e !important; color: #fff !important; }}
</style>
<div id="flt-drop">
  <div class="flt-head" id="flt-title">Filter</div>
  <div class="flt-search"><input type="text" id="flt-search" placeholder="Search values..." oninput="fltSearch()"></div>
  <div class="flt-items" id="flt-items"></div>
  <div class="flt-actions">
    <button onclick="fltAll()">All</button>
    <button onclick="fltNone()">None</button>
    <button class="flt-apply" onclick="fltApply()">Apply &#10003;</button>
  </div>
</div>
<script>
var _flt = {{ tid: null, col: null, btn: null }};
var _activeFilters = {{}}; // key: tid+'_'+col => Set of allowed values

function openFilter(e, tid, col) {{
  e.stopPropagation();
  var drop = document.getElementById('flt-drop');
  // If already open for same column, close
  if (drop.style.display === 'block' && _flt.tid === tid && _flt.col === col) {{
    drop.style.display = 'none'; return;
  }}
  _flt.tid = tid; _flt.col = col; _flt.btn = e.target;
  // Collect unique values from column
  var tbl = document.getElementById(tid);
  var rows = Array.from(tbl.querySelector('tbody').querySelectorAll('tr'));
  var vals = {{}};
  rows.forEach(function(r) {{
    var c = r.cells[col];
    var v = c ? c.innerText.trim() : '';
    vals[v] = (vals[v]||0) + 1;
  }});
  var sorted = Object.keys(vals).sort(function(a,b) {{
    var an = parseFloat(a), bn = parseFloat(b);
    if (!isNaN(an) && !isNaN(bn)) return an - bn;
    return a.localeCompare(b);
  }});
  var key = tid + '_' + col;
  var active = _activeFilters[key] || null;
  document.getElementById('flt-title').textContent = tbl.querySelectorAll('.sort-row th')[col].querySelector('.sort-label').textContent.trim();
  document.getElementById('flt-search').value = '';
  var items = document.getElementById('flt-items');
  items.innerHTML = sorted.map(function(v) {{
    var chk = (!active || active.has(v)) ? 'checked' : '';
    return '<label><input type="checkbox" value="' + encodeURIComponent(v) + '" ' + chk + '> ' + (v||'(empty)') + ' <span style="color:#aaa;font-size:.8em">(' + vals[v] + ')</span></label>';
  }}).join('');
  // Position dropdown
  var rect = e.target.getBoundingClientRect();
  drop.style.display = 'block';
  var dw = drop.offsetWidth;
  var left = Math.min(rect.left, window.innerWidth - dw - 8);
  drop.style.left = left + 'px';
  drop.style.top = (rect.bottom + 4) + 'px';
}}
document.addEventListener('click', function(e) {{
  var drop = document.getElementById('flt-drop');
  if (!drop.contains(e.target) && !e.target.classList.contains('filter-btn')) {{
    drop.style.display = 'none';
  }}
}});
function fltSearch() {{
  var q = document.getElementById('flt-search').value.toLowerCase();
  document.querySelectorAll('#flt-items label').forEach(function(l) {{
    l.style.display = l.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}
function fltAll() {{ document.querySelectorAll('#flt-items input[type=checkbox]').forEach(function(c){{c.checked=true;}}); }}
function fltNone() {{ document.querySelectorAll('#flt-items input[type=checkbox]').forEach(function(c){{c.checked=false;}}); }}
function fltApply() {{
  var key = _flt.tid + '_' + _flt.col;
  var checks = document.querySelectorAll('#flt-items input[type=checkbox]');
  var allChecked = Array.from(checks).every(function(c){{return c.checked;}});
  if (allChecked) {{
    delete _activeFilters[key];
  }} else {{
    var sel = new Set();
    checks.forEach(function(c){{ if(c.checked) sel.add(decodeURIComponent(c.value)); }});
    _activeFilters[key] = sel;
  }}
  // Mark filter button active/inactive
  var tbl = document.getElementById(_flt.tid);
  var btns = tbl.querySelectorAll('.filter-btn');
  btns[_flt.col] && btns[_flt.col].classList.toggle('active', !!_activeFilters[key]);
  applyAllFilters(_flt.tid);
  document.getElementById('flt-drop').style.display = 'none';
}}
function applyAllFilters(tid) {{
  var tbl = document.getElementById(tid);
  var rows = Array.from(tbl.querySelector('tbody').querySelectorAll('tr'));
  // Get all active filters for this table
  var filters = [];
  Object.keys(_activeFilters).forEach(function(k) {{
    if (k.startsWith(tid + '_')) {{
      var col = parseInt(k.split('_').pop());
      filters.push({{ col: col, vals: _activeFilters[k] }});
    }}
  }});
  rows.forEach(function(row) {{
    var show = filters.every(function(f) {{
      var c = row.cells[f.col];
      var v = c ? c.innerText.trim() : '';
      return f.vals.has(v);
    }});
    row.classList.toggle('filtered-out', !show);
  }});
}}
function sortTable(tid, col) {{
  var tbl = document.getElementById(tid);
  var tbody = tbl.querySelector('tbody');
  var rows = Array.from(tbody.querySelectorAll('tr'));
  var asc = tbl.dataset.sortCol == col && tbl.dataset.sortDir == 'asc';
  tbl.dataset.sortCol = col; tbl.dataset.sortDir = asc ? 'desc' : 'asc';
  rows.sort(function(a, b) {{
    var av = a.cells[col] ? a.cells[col].innerText.replace(/[%,]/g,'').trim() : '';
    var bv = b.cells[col] ? b.cells[col].innerText.replace(/[%,]/g,'').trim() : '';
    var an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return asc ? bn - an : an - bn;
    return asc ? bv.localeCompare(av) : av.localeCompare(bv);
  }});
  rows.forEach(function(r) {{ tbody.appendChild(r); }});
  var ths = tbl.querySelectorAll('.sort-row th .sort-icon');
  tbl.querySelectorAll('.sort-row th').forEach(function(th, i) {{
    var ic = th.querySelector('.sort-icon');
    if (ic) ic.textContent = (i == col) ? (asc ? '↑' : '↓') : '⇅';
  }});
}}
</script>
</head>
<body>

{LEGEND}

<div class="section">
  <h2>Top 10 runs by Accuracy</h2>
  {t1}
</div>

<div class="section">
  <h2>New Runs &nbsp;<span style="font-size:.82em;color:#999;font-weight:400">({log_note})</span></h2>
  {t2}
</div>

</body>
</html>"""

out = pathlib.Path('outputs/reports/results_report.html')
out.write_text(HTML, encoding='utf-8')
print(f'Saved: {out}  |  Table1: {len(table1)} rows  |  Table2: {log_note}')
