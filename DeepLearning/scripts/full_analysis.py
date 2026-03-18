"""
Full parameter analysis script.
Explores what different DM thresholds and DL hyperparameters would give.
"""
import pandas as pd
import numpy as np
import os
import sys

BASE = r"D:\MasterThesis\MasterThesis"

# ─────────────────────────────────────────────
# 1. STAGE1 QUALITY
# ─────────────────────────────────────────────
pq = pd.read_csv(os.path.join(BASE, "DataMining/results/reports/stage1/participant_quality.csv"))
tq = pd.read_csv(os.path.join(BASE, "DataMining/results/reports/stage1/trial_quality.csv"))
s3 = pd.read_csv(os.path.join(BASE, "DataMining/results/reports/stage3/stage3_with_labels.csv"))

print("=" * 70)
print("PARTICIPANT QUALITY (sorted by excluded_pct)")
print("=" * 70)
pq_sorted = pq[['participant_id', 'total_trials', 'excluded_trials', 'excluded_pct']].sort_values('excluded_pct')
print(pq_sorted.to_string(index=False))

print()
print("=" * 70)
print("EFFECT OF DIFFERENT PARTICIPANT EXCLUSION THRESHOLDS")
print("=" * 70)
for thresh in [0.10, 0.15, 0.20, 0.25, 0.33, 0.50]:
    kept = pq[pq['excluded_pct'] <= thresh]
    removed = pq[pq['excluded_pct'] > thresh]
    # How many trials/labels would remain in TC+TNC?
    tc_tnc = s3[
        s3['part'].isin(['Timer-Correct', 'Timer-No-Correct']) &
        s3['randomness_label'].isin(['RANDOM', 'NOT_RANDOM']) &
        s3['participant_id'].isin(kept['participant_id'])
    ]
    nr = (tc_tnc['randomness_label'] == 'NOT_RANDOM').sum()
    r  = (tc_tnc['randomness_label'] == 'RANDOM').sum()
    removed_ids = list(removed['participant_id'])
    print(f"  thresh={thresh:.2f}: keep {len(kept)}/30 participants, "
          f"TC+TNC={len(tc_tnc)} trials  [NR={nr}, R={r}]  "
          f"removed: {[p.replace('participant_','P') for p in sorted(removed_ids)]}")

# ─────────────────────────────────────────────
# 2. TRIAL QUALITY (invalid_pct)
# ─────────────────────────────────────────────
print()
print("=" * 70)
print("TRIAL INVALID_PCT DISTRIBUTION")
print("=" * 70)
desc = tq['invalid_pct'].describe(percentiles=[0.1, 0.2, 0.3, 0.5, 0.7])
print(desc.round(3))
print()
for t in [0.15, 0.20, 0.25, 0.30, 0.35]:
    passed = (tq['invalid_pct'] <= t).sum()
    print(f"  invalid_pct <= {t:.2f}: {passed}/{len(tq)} trials pass ({passed/len(tq)*100:.1f}%)")

# ─────────────────────────────────────────────
# 3. t_answer DISTRIBUTION (Timer-Correct only)
# ─────────────────────────────────────────────
print()
print("=" * 70)
print("t_answer DISTRIBUTION (Timer-Correct valid trials)")
print("=" * 70)
tc_valid = s3[(s3['part'] == 'Timer-Correct') & s3['t_answer'].notna()]
ta = tc_valid['t_answer']
print(f"  Count: {len(ta)}")
print(f"  min={ta.min():.1f}  P5={np.percentile(ta,5):.1f}  P10={np.percentile(ta,10):.1f}  "
      f"P15={np.percentile(ta,15):.1f}  P25={np.percentile(ta,25):.1f}  "
      f"P35={np.percentile(ta,35):.1f}  P50={np.percentile(ta,50):.1f}  "
      f"P75={np.percentile(ta,75):.1f}  max={ta.max():.1f}")
print()
for p in [10, 15, 20, 25, 30, 35, 40, 50]:
    thresh_val = np.percentile(ta, p)
    n_random = (ta <= thresh_val).sum()
    n_total_tnc = 136  # TNC always RANDOM
    total_random = n_total_tnc + n_random
    total_nr = len(ta) - n_random
    print(f"  P{p:2d} (t<={thresh_val:.1f}s): TC_random={n_random:3d}, "
          f"TOTAL_RANDOM={total_random} ({total_random/(total_random+total_nr)*100:.0f}%)  "
          f"NOT_RANDOM={total_nr}")

# ─────────────────────────────────────────────
# 4. ANSWER AREA SAMPLES per trial
# ─────────────────────────────────────────────
print()
print("=" * 70)
print("ANSWER_AREA samples per trial (min_answer_area_samples analysis)")
print("=" * 70)
# Load a few trial files to check
trials_dir = os.path.join(BASE, "DataMining/results/reports/final/trials")
if os.path.exists(trials_dir):
    aoi_counts = []
    tc_tnc_s3 = s3[s3['part'].isin(['Timer-Correct','Timer-No-Correct']) &
                   s3['randomness_label'].isin(['RANDOM','NOT_RANDOM'])]
    for _, row in tc_tnc_s3.iterrows():
        fpath = os.path.join(trials_dir, f"{row['participant_id']}_{row['question_id']}.csv")
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            # Count answer area samples in last 9s (1350 samples)
            df_tail = df.tail(1350)
            n_aoi = (df_tail.get('AOI', pd.Series(dtype=str)) == 'Answer_Area').sum()
            aoi_counts.append({'pid': row['participant_id'], 'qid': row['question_id'],
                                'label': row['randomness_label'], 'aoi_count': n_aoi})
    adf = pd.DataFrame(aoi_counts)
    if len(adf) > 0:
        ac = adf['aoi_count']
        print(f"  Count: {len(ac)} trials")
        print(f"  min={ac.min():.0f}  P5={np.percentile(ac,5):.0f}  P10={np.percentile(ac,10):.0f}  "
              f"P25={np.percentile(ac,25):.0f}  P50={np.percentile(ac,50):.0f}  "
              f"P75={np.percentile(ac,75):.0f}  max={ac.max():.0f}")
        print()
        for minval in [100, 200, 300, 400, 500]:
            passed = (ac >= minval).sum()
            print(f"  aoi_count >= {minval}: {passed}/{len(ac)} trials pass")
        print()
        # Distribution by label
        print("  AOI count by label:")
        for lbl, grp in adf.groupby('label'):
            ac_lbl = grp['aoi_count']
            print(f"    {lbl}: mean={ac_lbl.mean():.0f}  min={ac_lbl.min():.0f}  P10={np.percentile(ac_lbl,10):.0f}")

# ─────────────────────────────────────────────
# 5. WITHIN-PARTICIPANT CONSISTENCY (label coherence)
# ─────────────────────────────────────────────
print()
print("=" * 70)
print("WITHIN-PARTICIPANT LABEL COHERENCE")
print("=" * 70)
tc_tnc = s3[s3['part'].isin(['Timer-Correct','Timer-No-Correct']) &
            s3['randomness_label'].isin(['RANDOM','NOT_RANDOM'])]
print("Participants that are PURELY RANDOM (all trials = RANDOM):")
pure_r = []
mixed = []
for pid, grp in tc_tnc.groupby('participant_id'):
    nr = (grp['randomness_label']=='NOT_RANDOM').sum()
    r  = (grp['randomness_label']=='RANDOM').sum()
    if nr == 0:
        pure_r.append(f"{pid.replace('participant_','P')}({r}R)")
    elif nr <= 1:
        mixed.append(f"{pid.replace('participant_','P')}({nr}NR/{r}R)")
print(f"  Pure RANDOM: {pure_r}")
print(f"  Near-pure RANDOM (<=1 NR): {mixed}")
print()
print("If we remove pure-RANDOM participants:")
keep_pids = [pid for pid, grp in tc_tnc.groupby('participant_id')
             if (grp['randomness_label']=='NOT_RANDOM').sum() > 1]
sub = tc_tnc[tc_tnc['participant_id'].isin(keep_pids)]
nr_sub = (sub['randomness_label']=='NOT_RANDOM').sum()
r_sub  = (sub['randomness_label']=='RANDOM').sum()
print(f"  Keep {len(keep_pids)} participants → {len(sub)} trials  [NR={nr_sub}, R={r_sub}]")

print()
print("=" * 70)
print("SUMMARY: BEST PARAMETER CANDIDATES")
print("=" * 70)
