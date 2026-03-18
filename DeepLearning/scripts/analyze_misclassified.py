"""
Deep analysis of misclassified trials in the best LSTM run (seed=512, BAcc=68.9%).
Cross-references DL predictions with DataMining stage3 features to find patterns.

Run from: d:\MasterThesis\MasterThesis\DeepLearning\
"""
import json, os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, 'src')
from sklearn.model_selection import train_test_split

# â”€â”€ 1. Load best run â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
RUN = 'outputs/reports/lstm_20260306_181826.json'  # BAcc=68.9%, seed=512
d = json.load(open(RUN))
true_labels  = d['metrics']['test']['true_labels']
pred_labels  = d['metrics']['test']['predicted_labels']
probs        = d['metrics']['test'].get('predictions', [None]*len(true_labels))
seed         = d['data_config'].get('random_seed', 512)

# â”€â”€ 2. Load stage3 data (same sort as dataloader, includes t_answer & ta) â”€â”€â”€â”€
LABELS_CSV = '../DataMining/results/reports/stage3/stage3_with_labels.csv'
TRIALS_DIR = '../DataMining/results/reports/final/trials'

df = pd.read_csv(LABELS_CSV)
df = df[df['part'].isin(['Timer-Correct', 'Timer-No-Correct'])]
df = df[df['randomness_label'].isin(['RANDOM', 'NOT_RANDOM'])]

# Numeric sort (same as dataloader)
df['_pn'] = df['participant_id'].str.extract(r'(\d+)').astype(int)
df['_qn'] = df['question_id'].str.extract(r'(\d+)').astype(int)
df = df.sort_values(['_pn', '_qn']).drop(columns=['_pn','_qn']).reset_index(drop=True)

# Filter to valid trials (matching dataloader: skip if file missing or <10 AOI samples)
valid_rows = []
for _, row in df.iterrows():
    fname = f"{row['participant_id']}_{row['question_id']}.csv"
    fpath = os.path.join(TRIALS_DIR, fname)
    if os.path.exists(fpath):
        trial_df = pd.read_csv(fpath)
        aoi_rows = trial_df[trial_df['AOI'] == 'Answer_Area'] if 'AOI' in trial_df.columns else trial_df
        if len(aoi_rows) >= 10:
            valid_rows.append(row)

df_valid = pd.DataFrame(valid_rows).reset_index(drop=True)
y = (df_valid['randomness_label'] == 'RANDOM').astype(int).values

total = len(y)
print(f"Valid trials: {total} | NOT_RANDOM(0)={(y==0).sum()} | RANDOM(1)={(y==1).sum()}")

# â”€â”€ 3. Split same way as model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
idx = np.arange(total)
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)

df_test  = df_valid.iloc[idx_test].copy().reset_index(drop=True)
df_train = df_valid.iloc[idx_train].copy().reset_index(drop=True)
y_test   = y[idx_test]

# â”€â”€ 4. Attach predictions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
assert len(true_labels) == len(df_test), f"Mismatch: DL={len(true_labels)} vs df_test={len(df_test)}"
df_test['y_true']  = true_labels
df_test['y_pred']  = pred_labels
df_test['prob_RD'] = probs

def cat(row):
    if row['y_true']==0 and row['y_pred']==0: return 'TN'
    if row['y_true']==0 and row['y_pred']==1: return 'FP'
    if row['y_true']==1 and row['y_pred']==0: return 'FN'
    return 'TP'
df_test['cat'] = df_test.apply(cat, axis=1)

cm = df_test['cat'].value_counts()
print(f"\nCM: TN={cm.get('TN',0)} FP={cm.get('FP',0)} FN={cm.get('FN',0)} TP={cm.get('TP',0)}")
spec = cm.get('TN',0)/(cm.get('TN',0)+cm.get('FP',0))
rec  = cm.get('TP',0)/(cm.get('TP',0)+cm.get('FN',0))
print(f"BAcc={(spec+rec)/2*100:.1f}%  Spec={spec*100:.0f}%  Rec={rec*100:.0f}%")

# â”€â”€ 5. Per-trial full list â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'â”€'*90}")
print(f"{'Participant':<15} {'Question':<12} {'Part':<22} {'Label':<12} {'Pred':<6} {'t_answer':>9}  Outcome")
print('â”€'*90)
for _, row in df_test.sort_values(['cat','participant_id']).iterrows():
    label_name = 'NOT_RANDOM' if row['y_true']==0 else 'RANDOM'
    pred_name  = 'NR' if row['y_pred']==0 else 'RD'
    ta_val = f"{row['t_answer']:.1f}s" if pd.notna(row.get('t_answer')) else 'N/A'
    outcome = {'TN':'âœ“ TN','FP':'âœ— FP','FN':'âœ— FN','TP':'âœ“ TP'}[row['cat']]
    print(f"{row['participant_id']:<15} {row['question_id']:<12} {row['part']:<22} {label_name:<12} {pred_name:<6} {ta_val:>9}  {outcome}")

# â”€â”€ 6. Part distribution of errors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'='*60}")
print("ERRORS BY TRIAL TYPE:")
for part in ['Timer-Correct', 'Timer-No-Correct']:
    sub = df_test[df_test['part']==part]
    print(f"\n  {part} (n={len(sub)}):")
    print(f"    TN={sub['cat'].eq('TN').sum()}  FP={sub['cat'].eq('FP').sum()}  "
          f"FN={sub['cat'].eq('FN').sum()}  TP={sub['cat'].eq('TP').sum()}")
    if part == 'Timer-Correct':
        # Show t_answer for errors
        errs = sub[sub['cat'].isin(['FP','FN'])].sort_values('t_answer')
        if not errs.empty:
            print(f"    Errors (sorted by t_answer):")
            for _, r in errs.iterrows():
                print(f"      {r['cat']} {r['participant_id']} {r['question_id']} "
                      f"label={r['randomness_label']} t_answer={r['t_answer']:.1f}s  ta={r['ta']:.3f}s")

# â”€â”€ 7. t_answer distribution analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
tc = df_valid[df_valid['part']=='Timer-Correct'].copy()
threshold_p25 = np.percentile(tc['t_answer'], 25)
print(f"\n{'='*60}")
print(f"t_answer DISTRIBUTION (Timer-Correct only, n={len(tc)}):")
print(f"  P10={np.percentile(tc['t_answer'],10):.1f}s  P15={np.percentile(tc['t_answer'],15):.1f}s  "
      f"P20={np.percentile(tc['t_answer'],20):.1f}s  P25={np.percentile(tc['t_answer'],25):.1f}s  "
      f"P30={np.percentile(tc['t_answer'],30):.1f}s  P35={np.percentile(tc['t_answer'],35):.1f}s")
print(f"  Current threshold (P25): {threshold_p25:.1f}s")

# Show how many trials would flip labels at different percentiles
print(f"\n  Label impact at different percentiles:")
print(f"  {'Pct':>4} {'Threshold':>9} {'NOT_RANDOM':>11} {'RANDOM':>7} {'Delta_NR':>9}")
for pct in [10, 15, 20, 25, 30, 35, 40]:
    thr = np.percentile(tc['t_answer'], pct)
    nr = (tc['t_answer'] >= thr).sum()
    rd_tc = (tc['t_answer'] < thr).sum()
    delta = nr - (tc['t_answer'] >= threshold_p25).sum()
    print(f"  P{pct:<3} {thr:>8.1f}s {nr:>11} {rd_tc:>7}  {delta:>+9}")

# â”€â”€ 8. ta_window analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'='*60}")
print(f"ta (first fixation time on Answer_Area) DISTRIBUTION:")
print(f"  Timer-Correct:    min={tc['ta'].min():.2f}s  median={tc['ta'].median():.2f}s  max={tc['ta'].max():.2f}s")
tnc = df_valid[df_valid['part']=='Timer-No-Correct']
print(f"  Timer-No-Correct: min={tnc['ta'].min():.2f}s  median={tnc['ta'].median():.2f}s  max={tnc['ta'].max():.2f}s")

# â”€â”€ 9. Borderline trials (TC near threshold) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'='*60}")
print(f"BORDERLINE Timer-Correct TRIALS (t_answer within 5s of P25={threshold_p25:.1f}s):")
borderline = tc[abs(tc['t_answer'] - threshold_p25) < 5.0].sort_values('t_answer')
for _, r in borderline.iterrows():
    in_test = df_test[(df_test['participant_id']==r['participant_id']) & (df_test['question_id']==r['question_id'])]
    if not in_test.empty:
        outcome = in_test.iloc[0]['cat']
        print(f"  TEST  {r['participant_id']:<15} {r['question_id']:<12} t_answer={r['t_answer']:.1f}s  label={r['randomness_label']}  â†’ {outcome}")
    else:
        print(f"  TRAIN {r['participant_id']:<15} {r['question_id']:<12} t_answer={r['t_answer']:.1f}s  label={r['randomness_label']}")

