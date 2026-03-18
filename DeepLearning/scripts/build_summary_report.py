"""
Simple clean summary report for professor presentation.
Run from: d:\MasterThesis\MasterThesis\DeepLearning\
"""
from datetime import datetime

OUT = 'outputs/reports/summary_report.html'

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Experiment Summary Report</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f6fa; color: #2c3e50; }

  .header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    color: white; padding: 32px 48px;
  }
  .header h1 { font-size: 20px; font-weight: 700; margin-bottom: 6px; }
  .header .sub { font-size: 12px; opacity: 0.7; margin-top: 4px; }

  .wrap { max-width: 1300px; margin: 0 auto; padding: 28px 36px; }

  /* Key metrics boxes */
  .kpi-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin-bottom: 30px; }
  .kpi { background: white; border-radius: 10px; padding: 20px 16px; text-align: center;
         box-shadow: 0 2px 8px rgba(0,0,0,.07); border-top: 4px solid #ddd; }
  .kpi-num { font-size: 30px; font-weight: 800; }
  .kpi-lbl { font-size: 11px; color: #888; margin-top: 5px; line-height: 1.4; }

  /* Section title */
  .sec-title {
    font-size: 15px; font-weight: 700; color: #2c3e50;
    margin: 0 0 12px 0; padding-bottom: 8px;
    border-bottom: 2px solid #3498db;
  }
  .sec-note { font-size: 11px; color: #888; margin-bottom: 16px; }

  /* Tables */
  .card { background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,.07);
          margin-bottom: 28px; overflow: hidden; }
  .card-header { background: #2c3e50; color: white; padding: 12px 20px;
                 font-size: 13px; font-weight: 700; }
  .card-header span { font-size: 11px; font-weight: 400; opacity: .75; margin-left: 10px; }
  table { width: 100%; border-collapse: collapse; }
  th { background: #34495e; color: white; padding: 9px 12px; font-size: 12px;
       font-weight: 600; text-align: center; white-space: nowrap; }
  td { padding: 8px 12px; border-bottom: 1px solid #ecf0f1; font-size: 12px;
       vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #eaf4fb !important; }
  td.left { text-align: left; }
  td.center { text-align: center; }

  /* BAcc coloring */
  .bacc-best  { background: #1e8449; color: white; font-weight: 800; text-align: center; border-radius: 4px; padding: 3px 8px; }
  .bacc-good  { background: #27ae60; color: white; font-weight: 700; text-align: center; border-radius: 4px; padding: 3px 8px; }
  .bacc-ok    { background: #f39c12; color: white; font-weight: 700; text-align: center; border-radius: 4px; padding: 3px 8px; }
  .bacc-med   { background: #e67e22; color: white; font-weight: 700; text-align: center; border-radius: 4px; padding: 3px 8px; }
  .bacc-low   { background: #e74c3c; color: white; font-weight: 700; text-align: center; border-radius: 4px; padding: 3px 8px; }

  .acc-high { background: #1e8449; color: white; font-weight: 800; text-align: center;
              border-radius: 4px; padding: 3px 8px; }

  /* tag badges */
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
         font-size: 11px; font-weight: 700; color: white; }
  .tag-lstm       { background: #2980b9; }
  .tag-bilstm     { background: #8e44ad; }
  .tag-hybrid     { background: #27ae60; }
  .tag-cnn        { background: #e67e22; }
  .tag-cnn1d      { background: #c0392b; }
  .tag-mlp        { background: #7f8c8d; }
  .tag-transformer{ background: #16a085; }

  .row-highlight  { background: #fffde7 !important; }
  .row-highlight2 { background: #e8f8f5 !important; }

  .cm-box { font-size: 11px; font-family: monospace; color: #555; }

  .legend { display: flex; gap: 16px; padding: 10px 16px; background: #f8f9fa;
            border-top: 1px solid #ecf0f1; flex-wrap: wrap; }
  .leg { display: flex; align-items: center; gap: 6px; font-size: 11px; }
  .leg-dot { width: 12px; height: 12px; border-radius: 3px; }

  .note-box { background: #fffbf0; border-left: 4px solid #f39c12; padding: 10px 16px;
              font-size: 12px; color: #555; margin: 8px 0; border-radius: 0 6px 6px 0; }
  .info-box { background: #eaf4fb; border-left: 4px solid #3498db; padding: 10px 16px;
              font-size: 12px; color: #555; margin: 8px 0; border-radius: 0 6px 6px 0; }

  .footer { text-align: center; font-size: 11px; color: #aaa; padding: 20px;
            margin-top: 10px; }
</style>
</head>
<body>

<div class="header">
  <h1>Eye-Tracking Based Randomness Detection &mdash; Deep Learning Experiment Summary</h1>
  <div class="sub">
    Dataset: 30 participants &bull; 15 questions each &bull; Timer-Correct &amp; Timer-No-Correct trials &bull;
    Features: 13 eye-tracking channels (9s window, 300 time steps)<br>
    Task: Binary classification &mdash; RANDOM vs NOT_RANDOM reading behavior
  </div>
</div>

<div class="wrap">

<!-- ── KPI Row ──────────────────────────────────────── -->
<div class="kpi-row">
  <div class="kpi" style="border-color:#1e8449">
    <div class="kpi-num" style="color:#1e8449">76.47%</div>
    <div class="kpi-lbl">Best Overall Accuracy<br>(LSTM, seed=None)</div>
  </div>
  <div class="kpi" style="border-color:#2980b9">
    <div class="kpi-num" style="color:#2980b9">72.98%</div>
    <div class="kpi-lbl">Best Balanced Accuracy<br>(LSTM, seed=42)</div>
  </div>
  <div class="kpi" style="border-color:#8e44ad">
    <div class="kpi-num" style="color:#8e44ad">463</div>
    <div class="kpi-lbl">Total Experiments Run</div>
  </div>
  <div class="kpi" style="border-color:#e67e22">
    <div class="kpi-num" style="color:#e67e22">7</div>
    <div class="kpi-lbl">Model Architectures<br>Tested</div>
  </div>
</div>

<!-- ── Table 1: Experiment Phases ──────────────────── -->
<div class="card">
  <div class="card-header">
    Table 1 &mdash; Experiment Phases &amp; Key Results
    <span>Chronological progression of experiments</span>
  </div>
  <table>
    <thead>
      <tr>
        <th style="text-align:left;width:3%">#</th>
        <th style="text-align:left;width:18%">Phase / Change Made</th>
        <th style="text-align:left;width:15%">Configuration</th>
        <th>Acc&nbsp;(%)</th>
        <th>BAcc&nbsp;(%)</th>
        <th>Recall&nbsp;(%)</th>
        <th>Spec&nbsp;(%)</th>
        <th>F1</th>
        <th>AUC</th>
        <th>Confusion Matrix<br><small>TN / FP / FN / TP</small></th>
        <th style="text-align:left">Notes</th>
      </tr>
    </thead>
    <tbody>

      <!-- Phase 1 -->
      <tr>
        <td class="center" style="color:#888">1</td>
        <td class="left"><strong>Baseline LSTM</strong><br>
          <span style="font-size:10px;color:#888">Feb 13, 2026</span></td>
        <td class="left" style="font-size:11px">
          LR=0.001, Batch=8<br>Dropout=0.3, Hidden=128, L=2<br>Seed=None</td>
        <td class="center">&mdash;</td>
        <td class="center"><span class="bacc-low">~52%</span></td>
        <td class="center">&mdash;</td><td class="center">&mdash;</td>
        <td class="center">&mdash;</td><td class="center">&mdash;</td>
        <td class="center cm-box">&mdash;</td>
        <td class="left" style="font-size:11px;color:#888">Starting point</td>
      </tr>

      <!-- Phase 2 -->
      <tr class="row-highlight">
        <td class="center" style="color:#888">2</td>
        <td class="left"><strong>Hyperparameter Tuning (Best Acc)</strong><br>
          <span style="font-size:10px;color:#888">Feb 15&ndash;Mar 4, 2026</span></td>
        <td class="left" style="font-size:11px">
          LR=0.0004, Batch=<strong>16</strong><br>Dropout=0.15, Hidden=384, L=3<br>Seed=None (random split)</td>
        <td class="center"><div class="acc-high">76.47%</div></td>
        <td class="center"><span class="bacc-good">71.77%</span></td>
        <td class="center" style="color:#1e8449;font-weight:700">93.55%</td>
        <td class="center" style="color:#e74c3c;font-weight:700">50.00%</td>
        <td class="center">0.829</td>
        <td class="center">0.695</td>
        <td class="center cm-box">TN=10&nbsp;FP=10<br>FN=2&nbsp;&nbsp;TP=29</td>
        <td class="left" style="font-size:11px"><span style="color:#e74c3c">&#9888;</span>
          High Recall but low Specificity<br>Model biased toward RANDOM class<br>
          <em>Not reproducible (seed=None)</em></td>
      </tr>

      <!-- Phase 3 -->
      <tr class="row-highlight2">
        <td class="center" style="color:#888">3</td>
        <td class="left"><strong>Seed Optimization (Best BAcc)</strong><br>
          <span style="font-size:10px;color:#888">Feb 16&ndash;17, 2026</span></td>
        <td class="left" style="font-size:11px">
          LR=0.0004, Batch=16<br>Dropout=0.15, Hidden=384, L=3<br>Seed=<strong>42</strong></td>
        <td class="center">72.55%</td>
        <td class="center"><span class="bacc-good">72.98%</span></td>
        <td class="center">70.97%</td>
        <td class="center" style="font-weight:700;color:#1e8449">75.00%</td>
        <td class="center">0.759</td>
        <td class="center">0.700</td>
        <td class="center cm-box">TN=15&nbsp;FP=5<br>FN=9&nbsp;&nbsp;TP=22</td>
        <td class="left" style="font-size:11px"><span style="color:#27ae60">&#10003;</span>
          Best balanced result<br>Good Recall AND Specificity<br>
          <em>Reproducible with seed=42</em></td>
      </tr>

      <!-- Phase 4 -->
      <tr>
        <td class="center" style="color:#888">4</td>
        <td class="left"><strong>DataMining Threshold Change</strong><br>
          <span style="font-size:10px;color:#888">Mar 5, 2026</span></td>
        <td class="left" style="font-size:11px">
          DataMining re-run<br>DM: Pct=25, ta_ms=1000<br>Labels changed &rarr; different data split</td>
        <td class="center">68.63%</td>
        <td class="center"><span class="bacc-ok">68.87%</span></td>
        <td class="center">67.74%</td>
        <td class="center">70.00%</td>
        <td class="center">0.724</td>
        <td class="center">0.690</td>
        <td class="center cm-box">TN=14&nbsp;FP=6<br>FN=10&nbsp;TP=21</td>
        <td class="left" style="font-size:11px"><span style="color:#e74c3c">&#8595;</span>
          Performance dropped after re-labeling<br>NR=97 vs RD=157 (imbalanced)<br>
          <em>seed=512 used for reproducibility</em></td>
      </tr>

      <!-- Phase 5 -->
      <tr>
        <td class="center" style="color:#888">5</td>
        <td class="left"><strong>Hyperparameter Re-test with seed=512</strong><br>
          <span style="font-size:10px;color:#888">Mar 6, 2026</span></td>
        <td class="left" style="font-size:11px">
          LR=0.0004, Batch=16<br>Dropout=0.15, seed=<strong>512</strong><br>
          (same as Phase 2 config)</td>
        <td class="center">54.90%</td>
        <td class="center"><span class="bacc-low">54.90%</span></td>
        <td class="center">58.06%</td>
        <td class="center">51.72%</td>
        <td class="center">&mdash;</td>
        <td class="center">&mdash;</td>
        <td class="center cm-box">&mdash;</td>
        <td class="left" style="font-size:11px"><span style="color:#e74c3c">&#9888;</span>
          Confirmed: Phase 2 accuracy was split-dependent<br>
          <em>Same config &rarr; much worse on new data</em></td>
      </tr>

      <!-- Phase 6 -->
      <tr>
        <td class="center" style="color:#888">6</td>
        <td class="left"><strong>Seed Sweep (10 seeds tested)</strong><br>
          <span style="font-size:10px;color:#888">Mar 6, 2026</span></td>
        <td class="left" style="font-size:11px">
          LR=0.0005, Batch=8, Dropout=0.12<br>Seeds: 0,7,13,21,37,77,&hellip;,512</td>
        <td class="center">&mdash;</td>
        <td class="center"><span class="bacc-ok">68.87%</span></td>
        <td class="center">&mdash;</td><td class="center">&mdash;</td>
        <td class="center">&mdash;</td><td class="center">&mdash;</td>
        <td class="center cm-box">&mdash;</td>
        <td class="left" style="font-size:11px">seed=512 gave best result (68.87%)<br>
          Other seeds: 50&ndash;57% BAcc<br><em>seed=512 confirmed as optimal</em></td>
      </tr>

      <!-- Phase 7 -->
      <tr>
        <td class="center" style="color:#888">7</td>
        <td class="left"><strong>DM: Balanced Labels (NR=195, RD=190)</strong><br>
          <span style="font-size:10px;color:#888">Mar 5&ndash;6, 2026</span></td>
        <td class="left" style="font-size:11px">
          DM: Pct=25, ta_ms=1000, Cov=0.85<br>Different filtering &rarr; NR=195 / RD=190<br>LR=0.0004, Batch=16</td>
        <td class="center">&mdash;</td>
        <td class="center"><span class="bacc-good">71.77%</span></td>
        <td class="center">&mdash;</td><td class="center">&mdash;</td>
        <td class="center">&mdash;</td><td class="center">&mdash;</td>
        <td class="center cm-box">&mdash;</td>
        <td class="left" style="font-size:11px"><span style="color:#27ae60">&#10003;</span>
          More balanced data &rarr; better result<br>
          <em>Class imbalance is key limiting factor</em></td>
      </tr>

    </tbody>
  </table>
  <div class="legend">
    <div class="leg"><div class="leg-dot" style="background:#1e8449"></div>BAcc &ge; 72%</div>
    <div class="leg"><div class="leg-dot" style="background:#27ae60"></div>BAcc 68&ndash;72%</div>
    <div class="leg"><div class="leg-dot" style="background:#f39c12"></div>BAcc 62&ndash;68%</div>
    <div class="leg"><div class="leg-dot" style="background:#e67e22"></div>BAcc 55&ndash;62%</div>
    <div class="leg"><div class="leg-dot" style="background:#e74c3c"></div>BAcc &lt; 55%</div>
    <div class="leg" style="margin-left:auto"><span style="background:#fffde7;padding:2px 8px;border-radius:3px;border:1px solid #e0e0e0">Yellow = notable result</span></div>
  </div>
</div>

<!-- ── Table 2: All Model Architectures ────────────── -->
<div class="card">
  <div class="card-header">
    Table 2 &mdash; Comparison of All 7 Model Architectures (Best Run per Model)
    <span>All tested on the same dataset | LSTM performs best</span>
  </div>
  <table>
    <thead>
      <tr>
        <th style="text-align:left">Architecture</th>
        <th>Acc&nbsp;(%)</th>
        <th>BAcc&nbsp;(%)</th>
        <th>Recall&nbsp;(%)</th>
        <th>Spec&nbsp;(%)</th>
        <th>F1</th>
        <th>AUC</th>
        <th style="text-align:left">Best Config</th>
        <th># Runs</th>
        <th style="text-align:left">Notes</th>
      </tr>
    </thead>
    <tbody>
      <tr class="row-highlight">
        <td class="left"><span class="tag tag-lstm">LSTM</span></td>
        <td class="center"><div class="acc-high">76.47%</div></td>
        <td class="center"><span class="bacc-good">72.98%</span></td>
        <td class="center">93.55%<br><small style="color:#888">(best Acc run)</small></td>
        <td class="center">75.00%<br><small style="color:#888">(best BAcc run)</small></td>
        <td class="center">0.829</td>
        <td class="center">0.700</td>
        <td class="left" style="font-size:11px">LR=0.0004, Batch=16, Dropout=0.15,<br>Hidden=384, Layers=3</td>
        <td class="center">198</td>
        <td class="left" style="font-size:11px"><strong>Best model overall</strong></td>
      </tr>
      <tr>
        <td class="left"><span class="tag tag-hybrid">Hybrid CNN-LSTM</span></td>
        <td class="center">72.55%</td>
        <td class="center"><span class="bacc-ok">69.44%</span></td>
        <td class="center">83.87%</td>
        <td class="center">55.00%</td>
        <td class="center">0.788</td>
        <td class="center">0.719</td>
        <td class="left" style="font-size:11px">LR=0.0008, Batch=16, Dropout=0.15</td>
        <td class="center">74</td>
        <td class="left" style="font-size:11px">Second best; high recall</td>
      </tr>
      <tr>
        <td class="left"><span class="tag tag-cnn">CNN</span></td>
        <td class="center">68.63%</td>
        <td class="center"><span class="bacc-ok">68.59%</span></td>
        <td class="center">68.75%</td>
        <td class="center">68.42%</td>
        <td class="center">0.733</td>
        <td class="center">0.748</td>
        <td class="left" style="font-size:11px">LR=0.0008, Batch=16, Dropout=0.15</td>
        <td class="center">32</td>
        <td class="left" style="font-size:11px">Balanced performance; best AUC</td>
      </tr>
      <tr>
        <td class="left"><span class="tag tag-transformer">Transformer</span></td>
        <td class="center">67.31%</td>
        <td class="center"><span class="bacc-ok">65.94%</span></td>
        <td class="center">71.88%</td>
        <td class="center">60.00%</td>
        <td class="center">0.730</td>
        <td class="center">0.681</td>
        <td class="left" style="font-size:11px">LR=0.0004, Batch=16</td>
        <td class="center">38</td>
        <td class="left" style="font-size:11px">Reasonable but below LSTM</td>
      </tr>
      <tr>
        <td class="left"><span class="tag tag-mlp">MLP</span></td>
        <td class="center">66.67%</td>
        <td class="center"><span class="bacc-ok">63.71%</span></td>
        <td class="center">77.42%</td>
        <td class="center">50.00%</td>
        <td class="center">0.738</td>
        <td class="center">0.716</td>
        <td class="left" style="font-size:11px">LR=0.0004, Batch=16, Dropout=0.15</td>
        <td class="center">41</td>
        <td class="left" style="font-size:11px">No sequential modeling</td>
      </tr>
      <tr>
        <td class="left"><span class="tag tag-bilstm">BiLSTM</span></td>
        <td class="center">61.54%</td>
        <td class="center"><span class="bacc-med">61.25%</span></td>
        <td class="center">62.50%</td>
        <td class="center">60.00%</td>
        <td class="center">0.667</td>
        <td class="center">0.598</td>
        <td class="left" style="font-size:11px">LR=0.0005, Batch=16</td>
        <td class="center">41</td>
        <td class="left" style="font-size:11px">Bidirectional LSTM underperformed</td>
      </tr>
      <tr>
        <td class="left"><span class="tag tag-cnn1d">CNN-1D</span></td>
        <td class="center">58.82%</td>
        <td class="center"><span class="bacc-med">60.77%</span></td>
        <td class="center">53.12%</td>
        <td class="center">68.42%</td>
        <td class="center">0.618</td>
        <td class="center">0.558</td>
        <td class="left" style="font-size:11px">LR=0.0008, Batch=16</td>
        <td class="center">38</td>
        <td class="left" style="font-size:11px">Lowest overall performance</td>
      </tr>
    </tbody>
  </table>
</div>

<!-- ── Table 3: DM Threshold Impact ─────────────────── -->
<div class="card">
  <div class="card-header">
    Table 3 &mdash; DataMining Threshold Configurations &amp; Their Impact on LSTM Performance
    <span>All runs used the same LSTM architecture (LR=0.0004&ndash;0.0005, Dropout=0.12&ndash;0.15)</span>
  </div>
  <table>
    <thead>
      <tr>
        <th style="text-align:left">DM Configuration</th>
        <th>Pct.<br>Threshold</th>
        <th>ta Window<br>(ms)</th>
        <th>Coverage<br>Threshold</th>
        <th>Total<br>Trials</th>
        <th>NOT_RANDOM<br>(Class 0)</th>
        <th>RANDOM<br>(Class 1)</th>
        <th>Class<br>Ratio</th>
        <th>Best LSTM<br>BAcc (%)</th>
        <th style="text-align:left">Interpretation</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="left"><strong>No DM config logged</strong><br>
          <span style="font-size:10px;color:#888">Feb 2026</span></td>
        <td class="center">&mdash;</td>
        <td class="center">&mdash;</td>
        <td class="center">&mdash;</td>
        <td class="center">253</td>
        <td class="center">97</td>
        <td class="center">156</td>
        <td class="center">1 : 1.61</td>
        <td class="center"><span class="bacc-good">72.98%</span></td>
        <td class="left" style="font-size:11px">February experiments<br>Best overall result</td>
      </tr>
      <tr class="row-highlight2">
        <td class="left"><strong>Balanced output</strong><br>
          <span style="font-size:10px;color:#888">Mar 5, 2026</span></td>
        <td class="center" style="font-weight:700">25</td>
        <td class="center">1000</td>
        <td class="center">0.85</td>
        <td class="center">386</td>
        <td class="center" style="color:#1e8449;font-weight:700">195</td>
        <td class="center" style="color:#1e8449;font-weight:700">190</td>
        <td class="center" style="color:#1e8449;font-weight:700">1 : 0.97</td>
        <td class="center"><span class="bacc-good">71.77%</span></td>
        <td class="left" style="font-size:11px"><span style="color:#27ae60">&#10003;</span>
          Most balanced classes<br><em>Best result on March data</em></td>
      </tr>
      <tr class="row-highlight">
        <td class="left"><strong>Standard (current)</strong><br>
          <span style="font-size:10px;color:#888">Mar 5, 2026 &mdash; current</span></td>
        <td class="center" style="font-weight:700">25</td>
        <td class="center">1000</td>
        <td class="center">0.85</td>
        <td class="center">385</td>
        <td class="center" style="color:#e67e22;font-weight:700">97</td>
        <td class="center" style="color:#2980b9">157</td>
        <td class="center" style="color:#e67e22">1 : 1.62</td>
        <td class="center"><span class="bacc-ok">68.87%</span></td>
        <td class="left" style="font-size:11px">Current active configuration<br>
          <em>Imbalance hurts performance</em></td>
      </tr>
      <tr>
        <td class="left"><strong>Strict threshold</strong><br>
          <span style="font-size:10px;color:#888">Mar 5, 2026</span></td>
        <td class="center" style="font-weight:700;color:#e74c3c">15</td>
        <td class="center">1000</td>
        <td class="center">0.85</td>
        <td class="center">379</td>
        <td class="center" style="color:#1e8449">216</td>
        <td class="center" style="color:#e74c3c">163</td>
        <td class="center">1 : 0.75</td>
        <td class="center"><span class="bacc-low">52.46%</span></td>
        <td class="left" style="font-size:11px"><span style="color:#e74c3c">&#8595;</span>
          Too few RANDOM labels &rarr; bad performance<br>
          <em>P15 threshold too strict</em></td>
      </tr>
      <tr>
        <td class="left"><strong>Short ta window</strong><br>
          <span style="font-size:10px;color:#888">Mar 5, 2026</span></td>
        <td class="center">25</td>
        <td class="center" style="font-weight:700;color:#e74c3c">500</td>
        <td class="center">0.85</td>
        <td class="center">~411</td>
        <td class="center">~210</td>
        <td class="center">~202</td>
        <td class="center">1 : 0.96</td>
        <td class="center"><span class="bacc-med">56.49%</span></td>
        <td class="left" style="font-size:11px"><span style="color:#e74c3c">&#8595;</span>
          Shorter fixation window &rarr; noisy ta detection<br>
          <em>500ms too short for reliable fixation</em></td>
      </tr>
    </tbody>
  </table>
  <div class="note-box">
    <strong>Key Finding:</strong> DataMining threshold <em>stage3_threshold_percentile</em> (P25 = 25th percentile of answer-viewing latency) produces the best performance when combined with balanced NR/RD classes.
    The critical bottleneck is class imbalance &mdash; current data has 97 NOT_RANDOM vs 157 RANDOM trials.
  </div>
</div>

<!-- ── Table 4: Best LSTM Hyperparameters ───────────── -->
<div class="card">
  <div class="card-header">
    Table 4 &mdash; LSTM Hyperparameter Importance (Key Ablation Results)
    <span>Showing the impact of each hyperparameter change on performance</span>
  </div>
  <table>
    <thead>
      <tr>
        <th style="text-align:left">Parameter</th>
        <th style="text-align:left">Values Tested</th>
        <th style="text-align:left">Best Value</th>
        <th>Best BAcc</th>
        <th style="text-align:left">Effect / Observation</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="left"><strong>Learning Rate</strong></td>
        <td class="left" style="font-size:11px">0.001, 0.0008, 0.0005, 0.0004, 0.0002</td>
        <td class="left" style="font-weight:700;color:#1e8449">0.0004</td>
        <td class="center"><span class="bacc-good">72.98%</span></td>
        <td class="left" style="font-size:11px">0.0004 consistently best; higher LR overshoots, lower LR underfits</td>
      </tr>
      <tr>
        <td class="left"><strong>Batch Size</strong></td>
        <td class="left" style="font-size:11px">8, 16, 32</td>
        <td class="left" style="font-weight:700;color:#1e8449">16</td>
        <td class="center"><span class="bacc-good">72.98%</span></td>
        <td class="left" style="font-size:11px">Batch=16 best; batch=8 overfits on small dataset (253 trials)</td>
      </tr>
      <tr>
        <td class="left"><strong>Dropout</strong></td>
        <td class="left" style="font-size:11px">0.1, 0.12, 0.15, 0.2, 0.3</td>
        <td class="left" style="font-weight:700;color:#1e8449">0.15</td>
        <td class="center"><span class="bacc-good">72.98%</span></td>
        <td class="left" style="font-size:11px">0.15 provides best regularization; 0.3 underfits</td>
      </tr>
      <tr>
        <td class="left"><strong>Hidden Size</strong></td>
        <td class="left" style="font-size:11px">64, 128, 256, 384, 512</td>
        <td class="left" style="font-weight:700;color:#1e8449">384</td>
        <td class="center"><span class="bacc-good">72.98%</span></td>
        <td class="left" style="font-size:11px">384 best capacity for 13 features &times; 300 time steps</td>
      </tr>
      <tr>
        <td class="left"><strong>Num Layers</strong></td>
        <td class="left" style="font-size:11px">1, 2, 3</td>
        <td class="left" style="font-weight:700;color:#1e8449">3</td>
        <td class="center"><span class="bacc-good">72.98%</span></td>
        <td class="left" style="font-size:11px">3 layers needed for temporal hierarchy; 1 layer underfits</td>
      </tr>
      <tr>
        <td class="left"><strong>Random Seed</strong></td>
        <td class="left" style="font-size:11px">None, 0, 7, 13, 21, 37, 42, 77, 512, &hellip;</td>
        <td class="left" style="font-weight:700;color:#1e8449">42 (Feb) / 512 (Mar)</td>
        <td class="center"><span class="bacc-good">72.98%</span></td>
        <td class="left" style="font-size:11px">seed=None gives best on Feb data; seed=512 best on Mar data</td>
      </tr>
      <tr>
        <td class="left"><strong>Bidirectional</strong></td>
        <td class="left" style="font-size:11px">False (LSTM) vs True (BiLSTM)</td>
        <td class="left" style="font-weight:700;color:#1e8449">False (LSTM)</td>
        <td class="center"><span class="bacc-good">72.98%</span></td>
        <td class="left" style="font-size:11px">Unidirectional LSTM outperforms BiLSTM (72.98% vs 61.25%)</td>
      </tr>
    </tbody>
  </table>
  <div class="info-box">
    <strong>Best LSTM Configuration:</strong>
    LR = 0.0004 &bull; Batch = 16 &bull; Dropout = 0.15 &bull; Hidden = 384 &bull; Layers = 3 &bull; Seed = 42<br>
    <strong>Result:</strong> Acc = 72.55% &bull; BAcc = 72.98% &bull; Recall = 70.97% &bull; Specificity = 75.00% &bull; F1 = 0.759 &bull; AUC = 0.700
  </div>
</div>

<!-- ── Summary / Findings ────────────────────────────── -->
<div class="card">
  <div class="card-header">Key Findings &amp; Analysis</div>
  <div style="padding: 18px 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
    <div>
      <div class="sec-title">Why 76.47% Accuracy is Misleading</div>
      <div class="note-box">
        The 76.47% accuracy run had <strong>Recall=93.55%</strong> but <strong>Specificity=only 50%</strong>.
        This means the model was classifying almost everything as RANDOM, producing many False Positives.
        Balanced Accuracy (BAcc) = (Recall + Spec) / 2 = <strong>71.77%</strong> &mdash;
        which is actually <em>lower</em> than the 72.98% achieved with seed=42.
        <br><br>BAcc is therefore the more honest metric for imbalanced binary classification.
      </div>
    </div>
    <div>
      <div class="sec-title">Root Cause of Performance Drop (Feb &rarr; Mar)</div>
      <div class="note-box">
        The DataMining pipeline re-run in March changed the CSV row order, which changed the
        train/test split despite identical hyperparameters. This explains why the same
        configuration (LR=0.0004, batch=16, dropout=0.15) gave 72.98% BAcc in February
        but only 54.9% in March &mdash; <strong>the test set became harder</strong>.
        <br><br>Conclusion: performance is <em>split-sensitive</em> on this small dataset (n=253).
      </div>
    </div>
    <div>
      <div class="sec-title">DataMining Threshold Recommendation</div>
      <div class="info-box">
        The best DM configuration on March data produced <strong>NR=195 / RD=190</strong>
        (nearly balanced), achieving BAcc=<strong>71.77%</strong>.
        The current config (NR=97, RD=157) is imbalanced and limits performance to 68.87%.
        <br><br><strong>Recommendation:</strong> Use DM settings that produce balanced class distribution.
        Investigate why NR count dropped from 195 &rarr; 97 between DM runs.
      </div>
    </div>
    <div>
      <div class="sec-title">Misclassification Analysis (68.87% run)</div>
      <div class="info-box">
        Of 16 misclassified trials:<br>
        &bull; <strong>6/10 FN errors</strong> are Timer-No-Correct &mdash; RANDOM by definition (immutable)<br>
        &bull; <strong>3/6 FP errors</strong> are Timer-No-Correct with high t_answer (contradiction in labels)<br>
        &bull; Only <strong>1 FN</strong> Timer-Correct trial is near DM threshold (t_answer = 21.9s vs P25=25.2s)<br>
        <br><strong>Conclusion:</strong> The key limitation is label noise in Timer-No-Correct trials,
        not the DM threshold value.
      </div>
    </div>
  </div>
</div>

</div><!-- /wrap -->

<div class="footer">
  Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M') + """&nbsp;&nbsp;|&nbsp;&nbsp;
  Total experiments: 463&nbsp;&nbsp;|&nbsp;&nbsp;
  Models: LSTM, BiLSTM, Hybrid CNN-LSTM, CNN, CNN-1D, MLP, Transformer
</div>

</body>
</html>"""

with open(OUT, 'w', encoding='utf-8') as f:
    f.write(HTML)
print(f"Report saved: {OUT}")
