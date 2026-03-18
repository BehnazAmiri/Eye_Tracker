"""
Full experiment table: ONLY hyperparameters, thresholds, and results.
Run from: d:\MasterThesis\MasterThesis\DeepLearning\
"""
import pandas as pd, re, math
from datetime import datetime

CSV_IN = 'outputs/reports/experiment_report.csv'
OUT    = 'outputs/reports/full_table_report.html'

df = pd.read_csv(CSV_IN)

def parse_cm(s):
    if not isinstance(s, str): return '', '', '', ''
    m = re.findall(r'(\d+)', s)
    return (m[0], m[1], m[2], m[3]) if len(m) >= 4 else ('', '', '', '')

df[['TN','FP','FN','TP']] = df['CM'].apply(lambda x: pd.Series(parse_cm(x)))

def fmt(v, d=2):
    try:
        f = float(v)
        return '—' if math.isnan(f) else f'{f:.{d}f}'
    except:
        return '—' if str(v) in ('nan','None','') else str(v)

def fmtv(v):
    if pd.isna(v): return '—'
    s = str(v)
    return s[:-2] if s.endswith('.0') else s

def bacc_cls(v):
    try:
        f = float(v)
        if f >= 72: return 'g3'
        if f >= 68: return 'g2'
        if f >= 62: return 'g1'
        if f >= 55: return 'y1'
        return 'r1'
    except: return ''

MODEL_COLOR = {
    'LSTM':'#2980b9','BILSTM':'#8e44ad','HYBRID':'#27ae60',
    'CNN':'#e67e22','CNN1D':'#c0392b','MLP':'#7f8c8d','TRANSFORMER':'#16a085'
}
def model_badge(m):
    c = MODEL_COLOR.get(str(m).upper(), '#555')
    return f'<span class="badge" style="background:{c}">{m}</span>'

df['_bacc'] = pd.to_numeric(df['BAcc'], errors='coerce')
df = df.sort_values('_bacc', ascending=False).reset_index(drop=True)
best_bacc_idx = df['_bacc'].idxmax()
best_acc_idx  = pd.to_numeric(df['Acc'], errors='coerce').idxmax()

rows_html = []
for i, r in df.iterrows():
    hl = ''
    if i == best_bacc_idx: hl = 'row-best-bacc'
    elif i == best_acc_idx: hl = 'row-best-acc'
    bc = bacc_cls(r['BAcc'])
    has_dm = not pd.isna(r.get('DM_Pct', float('nan')))
    def dm_tag(v):
        val = fmtv(v)
        return f'<b>{val}</b>' if has_dm and val != chr(8212) else '<span class="nd">—</span>'

    rows_html.append(f"""<tr class="{hl}">
<td class="center">{i+1}</td>
<td class="center">{model_badge(r['Model'])}</td>
<td class="center hp">{fmt(r['LR'],6)}</td>
<td class="center hp">{fmtv(r['Batch'])}</td>
<td class="center hp">{fmt(r['Dropout'],2)}</td>
<td class="center hp">{fmtv(r['Hidden'])}</td>
<td class="center hp">{fmtv(r['Layers'])}</td>
<td class="center hp">{fmtv(r['Seed'])}</td>
<td class="center dm col-dm">{dm_tag(r['DM_Pct'])}</td>
<td class="center dm col-dm">{dm_tag(r['DM_ta_ms'])}</td>
<td class="center dm col-dm">{dm_tag(r['DM_Cov'])}</td>
<td class="center dm col-dm">{dm_tag(r['DM_Excl'])}</td>
<td class="center res">{fmt(r['Acc'],2)}%</td>
<td class="center res {bc}"><b>{fmt(r['BAcc'],2)}%</b></td>
<td class="center res">{fmt(r['Recall'],2)}%</td>
<td class="center res">{fmt(r['Spec'],2)}%</td>
<td class="center res">{fmt(r['F1'],3)}</td>
<td class="center res">{fmt(r['AUC'],3)}</td>
<td class="center cm col-cm">{r['TN']}</td>
<td class="center cm col-cm">{r['FP']}</td>
<td class="center cm col-cm">{r['FN']}</td>
<td class="center cm col-cm">{r['TP']}</td>
</tr>""")

rows_str = '\n'.join(rows_html)
total     = len(df)
generated = datetime.now().strftime('%Y-%m-%d %H:%M')

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment Results</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#f5f6fa;color:#2c3e50;font-size:12px}}
.hdr{{background:linear-gradient(135deg,#1a1a2e,#0f3460);color:white;padding:18px 28px}}
.hdr h1{{font-size:16px;font-weight:700}}
.hdr p{{font-size:11px;opacity:.7;margin-top:4px}}
.ctrl{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:white;
       padding:10px 16px;margin:14px 16px 10px;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,.07)}}
.ctrl label{{font-size:11px;font-weight:600;color:#555}}
.ctrl input,.ctrl select{{padding:4px 9px;border:1px solid #ddd;border-radius:4px;font-size:12px}}
.stat{{background:#ecf0f1;padding:3px 10px;border-radius:4px;font-size:11px}}
.legend{{display:flex;gap:14px;align-items:center;padding:0 16px 10px;flex-wrap:wrap;font-size:11px}}
.leg{{display:flex;align-items:center;gap:5px}}
.legd{{width:14px;height:14px;border-radius:3px}}
.wrap{{padding:0 16px 20px;overflow-x:auto}}
table{{width:100%;border-collapse:collapse;background:white;border-radius:8px;
       overflow:hidden;box-shadow:0 2px 10px rgba(0,0,0,.09);white-space:nowrap}}
.grp{{text-align:center;font-size:10px;font-weight:800;text-transform:uppercase;
      letter-spacing:.5px;padding:5px 4px;color:white}}
.grp-idx{{background:#607d8b}}.grp-hp{{background:#2c3e50}}
.grp-dm{{background:#1a5276}}.grp-res{{background:#1e8449}}.grp-cm{{background:#1b6ca8}}
th{{padding:7px 9px;font-size:11px;font-weight:700;text-align:center;color:white;
    border:1px solid rgba(255,255,255,.12);cursor:pointer;position:sticky;top:0;z-index:2;white-space:nowrap}}
th:hover{{opacity:.82}}
th.asc::after{{content:' ▲';font-size:9px}}th.dsc::after{{content:' ▼';font-size:9px}}
td{{padding:6px 9px;border-bottom:1px solid #ecf0f1;vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:#eaf4fb!important}}
.center{{text-align:center}}
.hp{{background:#f9f9fc!important}}.dm{{background:#eaf4fb!important}}
.cm{{font-family:monospace;font-size:11px;color:#777;background:#f8f8f8!important}}
.nd{{color:#ccc}}
.g3{{background:#d5f5e3!important;color:#1e8449}}
.g2{{background:#d6eaf8!important;color:#1a5276}}
.g1{{background:#fef9e7!important;color:#7d6608}}
.y1{{background:#fef5e7!important;color:#a04000}}
.r1{{background:#fdedec!important;color:#922b21}}
.row-best-bacc td{{border-left:4px solid #1e8449!important}}
.row-best-acc  td{{border-left:4px solid #f39c12!important}}
.badge{{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:700;color:white}}
tr.hide{{display:none}}
.footer{{text-align:center;font-size:11px;color:#aaa;padding:14px}}
.cbtn{{padding:3px 9px;border:1px solid #ddd;border-radius:4px;background:white;cursor:pointer;font-size:11px;font-weight:600}}
.cbtn.on{{background:#2c3e50;color:white;border-color:#2c3e50}}
</style>
</head>
<body>
<div class="hdr">
  <h1>Eye-Tracking — Deep Learning Experiment Results</h1>
  <p>{total} experiments &bull; Rows = Hyperparameters &amp; Thresholds &bull; Columns = Results &bull; Click column to sort</p>
</div>
<div class="ctrl">
  <label>Search:</label>
  <input id="q" placeholder="model, LR, seed..." oninput="filt()" style="width:170px">
  <label>Model:</label>
  <select id="mf" onchange="filt()">
    <option value="">All</option>
    <option>LSTM</option><option>BILSTM</option><option>HYBRID</option>
    <option>CNN</option><option>CNN1D</option><option>MLP</option><option>TRANSFORMER</option>
  </select>
  <label>Min BAcc (%):</label>
  <input id="mb" type="number" value="0" min="0" max="100" step="1" oninput="filt()" style="width:60px">
  <label style="display:flex;align-items:center;gap:4px;cursor:pointer;font-size:11px;font-weight:700;color:#e74c3c">
    <input type="checkbox" id="dmc" onchange="filt()"> DM runs only&nbsp;(57)
  </label>
  <div class="stat">Showing: <b id="vis">{total}</b> / {total}</div>
  <div style="margin-left:auto;display:flex;gap:6px">
    <button class="cbtn on" id="bdm" onclick="tog('dm')">DM Thresholds</button>
    <button class="cbtn on" id="bcm" onclick="tog('cm')">Confusion Matrix</button>
  </div>
</div>
<div class="legend">
  <b style="color:#555">BAcc:</b>
  <div class="leg"><div class="legd" style="background:#d5f5e3"></div>≥ 72%</div>
  <div class="leg"><div class="legd" style="background:#d6eaf8"></div>68–72%</div>
  <div class="leg"><div class="legd" style="background:#fef9e7"></div>62–68%</div>
  <div class="leg"><div class="legd" style="background:#fef5e7"></div>55–62%</div>
  <div class="leg"><div class="legd" style="background:#fdedec"></div>&lt;55%</div>
  <div class="leg" style="margin-left:12px"><div style="width:4px;height:14px;background:#1e8449;border-radius:2px"></div>&nbsp;Best BAcc</div>
  <div class="leg"><div style="width:4px;height:14px;background:#f39c12;border-radius:2px"></div>&nbsp;Best Acc</div>
  <span style="margin-left:auto;font-size:11px;color:#aaa">DM columns: <b style="color:#1a5276">bold</b> = value set &nbsp;|&nbsp; <span style="color:#ccc">—</span> = not logged (Feb 2026)</span>
</div>
<div class="wrap">
<table id="tbl">
<thead>
  <tr>
    <th rowspan="2" class="grp-idx" onclick="srt(0)">#</th>
    <th rowspan="2" class="grp-idx" onclick="srt(1)">Model</th>
    <th colspan="6" class="grp grp-hp">Hyperparameters</th>
    <th colspan="4" class="grp grp-dm col-dm">DataMining Thresholds</th>
    <th colspan="6" class="grp grp-res">Results</th>
    <th colspan="4" class="grp grp-cm col-cm">Confusion Matrix</th>
  </tr>
  <tr>
    <th class="grp-hp" onclick="srt(2)">LR</th>
    <th class="grp-hp" onclick="srt(3)">Batch</th>
    <th class="grp-hp" onclick="srt(4)">Dropout</th>
    <th class="grp-hp" onclick="srt(5)">Hidden</th>
    <th class="grp-hp" onclick="srt(6)">Layers</th>
    <th class="grp-hp" onclick="srt(7)">Seed</th>
    <th class="grp-dm col-dm" onclick="srt(8)">Pct</th>
    <th class="grp-dm col-dm" onclick="srt(9)">ta_ms</th>
    <th class="grp-dm col-dm" onclick="srt(10)">Coverage</th>
    <th class="grp-dm col-dm" onclick="srt(11)">Excl.</th>
    <th class="grp-res" onclick="srt(12)">Acc (%)</th>
    <th class="grp-res" onclick="srt(13)">BAcc (%)</th>
    <th class="grp-res" onclick="srt(14)">Recall (%)</th>
    <th class="grp-res" onclick="srt(15)">Spec (%)</th>
    <th class="grp-res" onclick="srt(16)">F1</th>
    <th class="grp-res" onclick="srt(17)">AUC</th>
    <th class="grp-cm col-cm" onclick="srt(18)">TN</th>
    <th class="grp-cm col-cm" onclick="srt(19)">FP</th>
    <th class="grp-cm col-cm" onclick="srt(20)">FN</th>
    <th class="grp-cm col-cm" onclick="srt(21)">TP</th>
  </tr>
</thead>
<tbody id="tb">
{rows_str}
</tbody>
</table>
</div>
<div class="footer">Generated: {generated} &bull; {total} experiments</div>
<script>
const dms={{show:true}}, cms={{show:true}};
function tog(g){{
  if(g==='dm'){{dms.show=!dms.show;document.querySelectorAll('.col-dm').forEach(e=>e.style.display=dms.show?'':'none');document.getElementById('bdm').classList.toggle('on',dms.show);}}
  else{{cms.show=!cms.show;document.querySelectorAll('.col-cm').forEach(e=>e.style.display=cms.show?'':'none');document.getElementById('bcm').classList.toggle('on',cms.show);}}
}}
function filt(){{
  const q=document.getElementById('q').value.toLowerCase();
  const mf=document.getElementById('mf').value.toUpperCase();
  const mb=parseFloat(document.getElementById('mb').value)||0;
  const dmc=document.getElementById('dmc').checked;
  let vis=0;
  document.querySelectorAll('#tb tr').forEach(row=>{{
    const txt=row.innerText.toLowerCase();
    const model=row.cells[1].innerText.trim().toUpperCase();
    const bacc=parseFloat(row.cells[13].innerText)||0;
    const hasDm=row.cells[8]&&row.cells[8].innerText.trim()!=='—';
    const ok=(!q||txt.includes(q))&&(!mf||model===mf)&&(bacc>=mb)&&(!dmc||hasDm);
    row.classList.toggle('hide',!ok);
    if(ok)vis++;
  }});
  document.getElementById('vis').textContent=vis;
}}
let sc=13,sd=-1;
function srt(col){{
  if(sc===col)sd*=-1;else{{sc=col;sd=-1;}}
  const rows=Array.from(document.querySelectorAll('#tb tr'));
  rows.sort((a,b)=>{{
    const av=a.cells[col]?.innerText.replace('%','').trim()||'';
    const bv=b.cells[col]?.innerText.replace('%','').trim()||'';
    const an=parseFloat(av),bn=parseFloat(bv);
    if(!isNaN(an)&&!isNaN(bn))return(an-bn)*sd;
    return av.localeCompare(bv)*sd;
  }});
  const tb=document.getElementById('tb');
  rows.forEach((r,i)=>{{tb.appendChild(r);r.cells[0].textContent=i+1;}});
}}
</script>
</body>
</html>"""

with open(OUT,'w',encoding='utf-8') as f:
    f.write(HTML)
print(f"Saved: {OUT} ({total} rows)")
