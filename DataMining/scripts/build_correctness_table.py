"""
Build Answer Correctness by Part & Randomness Label report.
Run from: d:\MasterThesis\MasterThesis\
Output:   DataMining/results/reports/answer_correctness_by_label.html
"""
import json, pathlib, pandas as pd

# ── Load answers and determine correctness ────────────────────────────────────
rows_ans = []
for ans_file in sorted(pathlib.Path('Eye_tracking_app/outputs').glob('participant_*/answers.json')):
    part_id = ans_file.parent.name
    answers = json.loads(ans_file.read_text(encoding='utf-8'))
    if isinstance(answers, dict):
        answers = [answers]
    for a in answers:
        qid = a.get('question_id') or a.get('question_number')
        chosen = a.get('chosen_option', '')
        is_correct = int(str(chosen).endswith('-C')) if chosen else 0
        rows_ans.append({
            'participant_id': part_id,
            'question_id': 'question_' + str(qid),
            'is_correct': is_correct
        })

answers_df = pd.DataFrame(rows_ans)

# ── Load stage3 trial-level data ──────────────────────────────────────────────
df = pd.read_csv('DataMining/results/reports/stage3/stage3_with_labels.csv')
df_trials = df.drop_duplicates(['participant_id', 'question_id'])
merged = df_trials.merge(answers_df, on=['participant_id', 'question_id'], how='left')
merged = merged[merged['randomness_label'].isin(['RANDOM', 'NOT_RANDOM'])]

# ── Build table rows ──────────────────────────────────────────────────────────
def make_rows(part_name):
    rows = []
    for label in ['RANDOM', 'NOT_RANDOM']:
        sub = merged[(merged['part'] == part_name) & (merged['randomness_label'] == label)]
        if len(sub) == 0:
            continue
        total   = len(sub)
        correct = int(sub['is_correct'].sum())
        incorrect = total - correct
        rows.append((label, total, correct, incorrect,
                     round(correct / total * 100, 1),
                     round(incorrect / total * 100, 1),
                     False))
    # TOTAL row
    sub = merged[merged['part'] == part_name]
    if len(sub):
        total   = len(sub)
        correct = int(sub['is_correct'].sum())
        incorrect = total - correct
        rows.append(('TOTAL', total, correct, incorrect,
                     round(correct / total * 100, 1),
                     round(incorrect / total * 100, 1),
                     True))
    return rows

# ── Render HTML rows ──────────────────────────────────────────────────────────
def label_cell(label):
    if label == 'RANDOM':
        return f'<td class="label-random">{label}</td>'
    elif label == 'NOT_RANDOM':
        return f'<td class="label-notrandom">{label}</td>'
    else:
        return f'<td><strong>{label}</strong></td>'

def render_part(part_display, part_key, rowspan, first_row_rendered, note=''):
    rows = make_rows(part_key)
    html = ''
    for i, (label, total, correct, incorrect, cp, ip, is_total) in enumerate(rows):
        tr_class = 'total-row' if is_total else ''
        html += f'<tr class="{tr_class}">'
        if i == 0:
            n = note and f'<br><small style="color:#888;font-style:italic">{note}</small>' or ''
            html += f'<td class="part-cell" rowspan="{rowspan}">{part_display}{n}</td>'
        html += label_cell(label)
        html += f'<td>{total}</td>'
        c_val = f'<strong>{correct}</strong>' if is_total else str(correct)
        i_val = f'<strong>{incorrect}</strong>' if is_total else str(incorrect)
        p_c   = f'<strong>{cp}%</strong>' if is_total else f'{cp}%'
        p_i   = f'<strong>{ip}%</strong>' if is_total else f'{ip}%'
        html += f'<td class="correct-col">{c_val}</td>'
        html += f'<td class="incorrect-col">{i_val}</td>'
        html += f'<td class="correct-col">{p_c}</td>'
        html += f'<td class="incorrect-col">{p_i}</td>'
        html += '</tr>\n'
    return html

# Count rowspan per part
def rowspan(part_key):
    n = sum(1 for lbl in ['RANDOM','NOT_RANDOM']
            if len(merged[(merged['part']==part_key) & (merged['randomness_label']==lbl)]) > 0)
    return n + 1  # +1 for TOTAL row

rs_notimer  = rowspan('No-Timer')
rs_timer_c  = rowspan('Timer-Correct')
rs_timer_nc = rowspan('Timer-No-Correct')

table_body = ''
table_body += render_part('No-Timer',          'No-Timer',          rs_notimer,  True)
table_body += f'<tr><td colspan="7" style="padding:0;background:#f0f0f0;height:6px"></td></tr>\n'
table_body += render_part('Timer-Correct',     'Timer-Correct',     rs_timer_c,  True)
table_body += f'<tr><td colspan="7" style="padding:0;background:#f0f0f0;height:6px"></td></tr>\n'
table_body += render_part('Timer-No-Correct',  'Timer-No-Correct',  rs_timer_nc, True,
                           note='no correct answers by design')

# ── DM threshold info ─────────────────────────────────────────────────────────
tc_json = pathlib.Path('DataMining/results/reports/threshold_comparison.json')
dm_info = ''
if tc_json.exists():
    tc = json.loads(tc_json.read_text(encoding='utf-8'))
    if isinstance(tc, list) and tc:
        best = tc[0]
        pct  = best.get('stage3_threshold_percentile', '?')
        ta   = best.get('ta_window_ms', '?')
        dm_info = f'Randomness threshold: answer time &lt; <strong>{pct}th percentile</strong> of all answer times &nbsp;|&nbsp; TA window: <strong>{ta} ms</strong>'

# ── Full HTML ─────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Answer Correctness by Part &amp; Randomness Label</title>
<style>
  body {{ font-family: Segoe UI, Arial, sans-serif; background: #f5f5f5; padding: 30px; color: #222; }}
  .card {{ background: #fff; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,.10); padding: 28px 32px; max-width: 860px; margin: auto; }}
  h2 {{ margin-top: 0; color: #4a3080; display: flex; align-items: center; gap: 10px; }}
  h2::before {{ content: "✅"; }}
  .subtitle {{ color: #444; margin: 6px 0 4px; }}
  .note {{ color: #888; font-style: italic; font-size: 0.88em; margin-bottom: 18px; }}
  .dm-info {{ background: #f0eaf8; border-left: 4px solid #7c5bbf; padding: 8px 14px;
              border-radius: 4px; font-size: 0.9em; margin-bottom: 20px; color: #333; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.95em; }}
  thead tr {{ background: #5b4095; color: #fff; }}
  thead th {{ padding: 11px 14px; text-align: left; font-weight: 600; }}
  tbody tr {{ border-bottom: 1px solid #e8e8e8; }}
  tbody tr:hover {{ background: #faf7ff; }}
  tbody tr.total-row {{ background: #f5f0ff; font-weight: 600; }}
  tbody tr.total-row:hover {{ background: #ede6ff; }}
  td {{ padding: 9px 14px; vertical-align: middle; }}
  .part-cell {{ color: #5b4095; font-weight: 600; font-size: 0.97em; }}
  .label-random    {{ color: #e07b00; font-weight: 700; }}
  .label-notrandom {{ color: #2a8a2a; font-weight: 700; }}
  .correct-col   {{ color: #2a8a2a; }}
  .incorrect-col {{ color: #c0392b; }}
</style>
</head>
<body>
<div class="card">
  <h2>Answer Correctness by Part &amp; Randomness Label</h2>
  <p class="subtitle">For each condition, how many trials in each randomness category were answered correctly vs incorrectly?</p>
  <p class="note">INVALID trials excluded. Timer-No-Correct has no correct answers by design.</p>
  {'<div class="dm-info">' + dm_info + '</div>' if dm_info else ''}
  <table>
    <thead>
      <tr>
        <th>Part</th><th>Label</th><th>Total Trials</th>
        <th>Correct</th><th>Incorrect</th><th>Correct %</th><th>Incorrect %</th>
      </tr>
    </thead>
    <tbody>
{table_body}
    </tbody>
  </table>
</div>
</body>
</html>
"""

out = pathlib.Path('DataMining/results/reports/answer_correctness_by_label.html')
out.write_text(html, encoding='utf-8')
print(f'Saved: {out}')
