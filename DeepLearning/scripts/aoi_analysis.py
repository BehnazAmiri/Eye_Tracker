import pandas as pd, numpy as np, os

trials_dir = r"D:\MasterThesis\MasterThesis\DataMining\results\reports\final\trials"
s3 = pd.read_csv(r"D:\MasterThesis\MasterThesis\DataMining\results\reports\stage3\stage3_with_labels.csv")
tc_tnc = s3[s3['part'].isin(['Timer-Correct','Timer-No-Correct']) & s3['randomness_label'].isin(['RANDOM','NOT_RANDOM'])]

rows = []
for _, row in tc_tnc.iterrows():
    pid = row['participant_id']
    qid = row['question_id']
    fp = os.path.join(trials_dir, f'{pid}_{qid}.csv')
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        aoi_col = df.get('AOI', pd.Series(dtype=str))
        n_aoi = (df.tail(1350)['AOI'] == 'Answer_Area').sum() if 'AOI' in df.columns else 0
        rows.append({'label': row['randomness_label'], 'part': row['part'], 'aoi': n_aoi, 'pid': pid})

df2 = pd.DataFrame(rows)
print("=== LOW AOI TRIALS (< 200 Answer_Area samples in last 9s) ===")
low = df2[df2['aoi'] < 200].sort_values('aoi')
print(f"Total low-AOI trials: {len(low)}")
for _, r in low.iterrows():
    print(f"  {r['pid']:15s}  part={r['part']:20s}  label={r['label']:12s}  aoi={r['aoi']:4.0f}")

print()
print("=== Removing low-AOI (< 200) class balance ===")
clean = df2[df2['aoi'] >= 200]
nr_clean = (clean['label']=='NOT_RANDOM').sum()
r_clean  = (clean['label']=='RANDOM').sum()
print(f"Before: {(df2['label']=='NOT_RANDOM').sum()} NR + {(df2['label']=='RANDOM').sum()} R  ({(df2['label']=='RANDOM').sum()/len(df2)*100:.1f}% RANDOM)")
print(f"After:  {nr_clean} NR + {r_clean} R = {len(clean)} total  ({r_clean/len(clean)*100:.1f}% RANDOM)")
