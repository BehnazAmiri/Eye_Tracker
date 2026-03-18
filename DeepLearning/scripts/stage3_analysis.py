import pandas as pd
import numpy as np

s3 = pd.read_csv('DataMining/results/reports/stage3/stage3_with_labels.csv')
tc = s3[s3['part'].isin(['Timer-Correct','Timer-No-Correct'])]

print('=== TC+TNC breakdown ===')
for part in ['Timer-Correct','Timer-No-Correct']:
    sub = tc[tc['part']==part]
    labels = sub['randomness_label'].value_counts()
    print(f'{part}: total={len(sub)},  labels={dict(labels)}')
print()

tnc = tc[tc['part']=='Timer-No-Correct']
print(f'Timer-No-Correct: {len(tnc)} trials => ALL RANDOM (hardcoded stage3.py line 68)')

tca = tc[tc['part']=='Timer-Correct']
print(f'Timer-Correct: {len(tca)} trials')
print(f'  Labels: {dict(tca["randomness_label"].value_counts())}')
print()

ta_valid = tca[tca['randomness_label'].isin(['RANDOM','NOT_RANDOM'])].dropna(subset=['t_answer'])
print('t_answer distribution (Timer-Correct only, seconds remaining):')
for p in [10,15,20,25,30,35,40,50]:
    v = np.percentile(ta_valid['t_answer'], p)
    n_rand = (ta_valid['t_answer'] < v).sum()
    print(f'  P{p:2d} = {v:.1f}s: {n_rand} TC classified RANDOM')
print()

tc_r = ta_valid[ta_valid['randomness_label']=='RANDOM']['t_answer']
tc_nr = ta_valid[ta_valid['randomness_label']=='NOT_RANDOM']['t_answer']
print(f'Current RANDOM TC: n={len(tc_r)}, mean={tc_r.mean():.1f}s, max={tc_r.max():.1f}s')
print(f'Current NOT_RANDOM TC: n={len(tc_nr)}, mean={tc_nr.mean():.1f}s, min={tc_nr.min():.1f}s')
print()
print(f'Overall (for DL training):')
print(f'  TNC all RANDOM: {len(tnc)}')
print(f'  TC RANDOM: {len(tc_r)}  NOT_RANDOM: {len(tc_nr)}')
total_random = len(tnc) + len(tc_r)
total_not = len(tc_nr)
total = total_random + total_not
print(f'  GRAND TOTAL => RANDOM: {total_random} ({100*total_random/total:.0f}%), NOT_RANDOM: {total_not} ({100*total_not/total:.0f}%)')
