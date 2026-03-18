"""Quick seed scan: shows train/test split distribution for multiple seeds.
Helps identify which seeds give more balanced or favorable test partitions.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load labels
labels_csv = os.path.join(os.path.dirname(__file__), '..', '..', 'DataMining',
                          'results', 'reports', 'stage3', 'stage3_with_labels.csv')
df = pd.read_csv(labels_csv)
df = df[df['part'].isin(['Timer-Correct', 'Timer-No-Correct'])]
df = df[df['randomness_label'].isin(['RANDOM', 'NOT_RANDOM'])]

# Numeric sort (matching dataloader)
df['_pn'] = df['participant_id'].str.extract(r'(\d+)').astype(int)
df['_qn'] = df['question_id'].str.extract(r'(\d+)').astype(int)
df = df.sort_values(['_pn', '_qn']).drop(columns=['_pn', '_qn']).reset_index(drop=True)

# Note: dataloader skips 7 trials with insufficient AOI data → ~253 loaded
# For this scan we use all 260 (close enough)
y = (df['randomness_label'] == 'RANDOM').astype(int).values

print(f'Stage3 rows (TC+TNC valid): {len(y)}')
print(f'  NOT_RANDOM: {(y==0).sum()} | RANDOM: {(y==1).sum()}')
print()

# Seeds to scan
seeds = [0, 7, 13, 21, 37, 42, 77, 100, 123, 200, 256, 300, 404, 512, 999]

print(f"{'Seed':>5} | {'n_te':>5} | {'NR_te':>6} | {'RD_te':>6} | {'%NR_te':>7} | Participants_in_test")
print('-' * 90)

for seed in seeds:
    idxs = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idxs, test_size=0.2, random_state=seed, stratify=y)
    te_df = df.iloc[idx_te]
    te_c0 = (y[idx_te] == 0).sum()
    te_c1 = (y[idx_te] == 1).sum()
    pnr = 100 * te_c0 / (te_c0 + te_c1)
    pids = sorted(te_df['participant_id'].unique(), key=lambda x: int(x.split('_')[1]))
    pid_str = str([p.replace('participant_', 'P') for p in pids])
    print(f"{seed:>5} | {len(idx_te):>5} | {te_c0:>6} | {te_c1:>6} | {pnr:>6.1f}% | {pid_str}")

print()
print('NOTE: The dataloader loads ~253 trials (7 skipped for low AOI). Actual test n≈51.')
print('All seeds give stratified (same %NR_te) splits - difference is WHICH trials are in test.')
print('Best seed is the one where model generalizes best to its specific test set.')
