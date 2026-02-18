"""
Stage 1: Quality Filtering using BPOGV and BKID
"""

import pandas as pd
import os


def run_stage1(df, config, report_dir='results/reports'):
    """
    Stage 1: Rule-based quality filtering.
    
    Validity Rules (sequential check):
      1. If BPOGV == 1 → Sample is VALID (gaze point is valid, don't check BKID)
      2. If BPOGV == 0 → Check BKID:
         - If BKID != 0 → Sample is VALID (blink detected, kept)
         - If BKID == 0 → Sample is INVALID (no valid gaze, no blink)
      
      Equivalent logic: (BPOGV == 1) OR (BKID != 0)
    
    Trial exclusion: if invalid_pct > 30% -> exclude entire trial
    Participant exclusion: if excluded_trials > 50% -> exclude entire participant
    
    Note: Individual samples are NOT deleted, only marked. Entire trials are excluded.
    """
    
    print("\n" + "="*80)
    print("STAGE 1: QUALITY FILTERING")
    print("="*80)
    
    # Get thresholds from config
    trial_threshold = config.getfloat('Analysis', 'stage1_invalid_pct_threshold', fallback=0.30)
    participant_threshold = config.getfloat('Analysis', 'stage1_participant_exclusion_threshold', fallback=0.50)
    
    print(f"  Trial threshold: {trial_threshold:.0%} invalid samples")
    print(f"  Participant threshold: {participant_threshold:.0%} excluded trials")
    
    # Check required columns
    required = ['BPOGV', 'BKID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Copy data
    df = df.copy()
    
    # Handle NaN values
    # BPOGV: -1 indicates invalid/missing gaze data
    # BKID: 0 indicates no blink (eyes open) - treat missing as 0 (conservative: assume no blink)
    df['BPOGV'] = df['BPOGV'].fillna(-1).astype('int8')
    df['BKID'] = df['BKID'].fillna(0).astype('int8')
    
    # Apply validity rules (sequential check):
    # Step 1: Initialize all as invalid
    df['is_valid'] = False
    
    # Step 2: If BPOGV == 1 → Valid (don't need to check BKID)
    df.loc[df['BPOGV'] == 1, 'is_valid'] = True
    
    # Step 3: If BPOGV != 1 (i.e., BPOGV == 0), check BKID:
    #         If BKID != 0 → Valid (blink detected, keep it)
    #         If BKID == 0 → Invalid (already False from Step 1)
    df.loc[(df['BPOGV'] != 1) & (df['BKID'] != 0), 'is_valid'] = True
    
    # Sample-level stats with breakdown
    total_samples = len(df)
    valid_samples = df['is_valid'].sum()
    invalid_samples = total_samples - valid_samples
    
    # Detailed validity breakdown
    bpogv_valid = (df['BPOGV'] == 1).sum()  # Valid gaze point (always valid)
    bkid_nonzero = (df['BKID'] != 0).sum()  # BKID != 0 (blink or other non-zero)
    both_conditions = ((df['BPOGV'] == 1) & (df['BKID'] != 0)).sum()  # Valid gaze AND BKID != 0 (overlap)
    neither_condition = ((df['BPOGV'] != 1) & (df['BKID'] == 0)).sum()  # Invalid: BPOGV=0 AND BKID=0
    
    print(f"\n  [SAMPLE-LEVEL VALIDITY]")
    print(f"    Total samples: {total_samples:,}")
    print(f"    Valid samples: {valid_samples:,} ({valid_samples/total_samples:.1%})")
    print(f"      - BPOGV=1 (valid gaze): {bpogv_valid:,} ({bpogv_valid/total_samples:.1%})")
    print(f"      - BKID≠0 (when BPOGV=0): {bkid_nonzero:,} ({bkid_nonzero/total_samples:.1%})")
    print(f"      - Overlap (BPOGV=1 & BKID≠0): {both_conditions:,} ({both_conditions/total_samples:.1%})")
    print(f"    Invalid samples (BPOGV=0 & BKID=0): {invalid_samples:,} ({invalid_samples/total_samples:.1%})")
    
    # Trial-level stats
    trial_stats = df.groupby(['participant_id', 'question_id', 'part']).agg(
        total_samples=('is_valid', 'size'),
        valid_samples=('is_valid', 'sum')
    ).reset_index()
    
    trial_stats['invalid_samples'] = trial_stats['total_samples'] - trial_stats['valid_samples']
    trial_stats['invalid_pct'] = trial_stats['invalid_samples'] / trial_stats['total_samples']
    trial_stats['excluded'] = trial_stats['invalid_pct'] > trial_threshold
    trial_stats['status'] = trial_stats['excluded'].map({True: 'EXCLUDED', False: 'KEPT'})
    
    total_trials = len(trial_stats)
    excluded_trials = trial_stats['excluded'].sum()
    kept_trials = total_trials - excluded_trials
    
    # Breakdown by threshold ranges
    trials_0_10 = ((trial_stats['invalid_pct'] >= 0) & (trial_stats['invalid_pct'] < 0.10)).sum()
    trials_10_20 = ((trial_stats['invalid_pct'] >= 0.10) & (trial_stats['invalid_pct'] < 0.20)).sum()
    trials_20_30 = ((trial_stats['invalid_pct'] >= 0.20) & (trial_stats['invalid_pct'] < 0.30)).sum()
    trials_30_plus = (trial_stats['invalid_pct'] >= 0.30).sum()
    
    print(f"\n  [TRIAL-LEVEL FILTERING]")
    print(f"    Threshold: Invalid > {trial_threshold:.0%}")
    print(f"    Total trials: {total_trials}")
    print(f"    Kept trials: {kept_trials} ({kept_trials/total_trials:.1%})")
    print(f"    Excluded trials: {excluded_trials} ({excluded_trials/total_trials:.1%})")
    print(f"\n    Invalid % Distribution:")
    print(f"      0-10%: {trials_0_10} trials")
    print(f"      10-20%: {trials_10_20} trials")
    print(f"      20-30%: {trials_20_30} trials")
    print(f"      >30% (excluded): {trials_30_plus} trials")
    
    # Participant-level stats
    participant_stats = trial_stats.groupby('participant_id').agg(
        total_trials=('excluded', 'size'),
        excluded_trials=('excluded', 'sum')
    ).reset_index()
    
    participant_stats['kept_trials'] = participant_stats['total_trials'] - participant_stats['excluded_trials']
    participant_stats['excluded_pct'] = participant_stats['excluded_trials'] / participant_stats['total_trials']
    participant_stats['participant_excluded'] = participant_stats['excluded_pct'] > participant_threshold
    participant_stats['status'] = participant_stats['participant_excluded'].map({True: 'EXCLUDED', False: 'KEPT'})
    
    # Mark all trials from excluded participants
    excluded_participants = participant_stats[participant_stats['participant_excluded']]['participant_id'].tolist()
    if excluded_participants:
        trial_stats.loc[trial_stats['participant_id'].isin(excluded_participants), 'excluded'] = True
        trial_stats.loc[trial_stats['participant_id'].isin(excluded_participants), 'status'] = 'EXCLUDED (participant)'
    
    total_participants = len(participant_stats)
    excluded_participants_count = participant_stats['participant_excluded'].sum()
    kept_participants = total_participants - excluded_participants_count
    
    print(f"\n  [PARTICIPANT-LEVEL FILTERING]")
    print(f"    Threshold: Excluded trials > {participant_threshold:.0%}")
    print(f"    Total participants: {total_participants}")
    print(f"    Kept participants: {kept_participants} ({kept_participants/total_participants:.1%})")
    print(f"    Excluded participants: {excluded_participants_count} ({excluded_participants_count/total_participants:.1%})")
    
    if excluded_participants_count > 0:
        print(f"\n    Excluded participants details:")
        for pid in excluded_participants:
            p_row = participant_stats[participant_stats['participant_id'] == pid].iloc[0]
            print(f"      {pid}: {p_row['excluded_trials']}/{p_row['total_trials']} trials excluded ({p_row['excluded_pct']:.1%})")
    
    # Recalculate final trial counts after participant exclusion
    final_excluded_trials = trial_stats['excluded'].sum()
    final_kept_trials = total_trials - final_excluded_trials
    
    print(f"\n  [FINAL TRIAL COUNTS]")
    print(f"    Total trials: {total_trials}")
    print(f"    Excluded (trial-level): {excluded_trials}")
    print(f"    Excluded (participant-level): {final_excluded_trials - excluded_trials}")
    print(f"    Total excluded: {final_excluded_trials} ({final_excluded_trials/total_trials:.1%})")
    print(f"    Kept for Stage 2: {final_kept_trials} ({final_kept_trials/total_trials:.1%})")
    
    # Filter dataframe to keep only accepted trials
    accepted_trials = trial_stats[~trial_stats['excluded']][['participant_id', 'question_id']]
    df_clean = df.merge(accepted_trials, on=['participant_id', 'question_id'], how='inner')
    
    print(f"\n  Final:")
    print(f"    Samples kept: {len(df_clean):,} / {total_samples:,} ({len(df_clean)/total_samples:.1%})")
    
    # Save audit files
    try:
        stage1_dir = os.path.join(report_dir, 'stage1')
        os.makedirs(stage1_dir, exist_ok=True)
        
        trial_stats.to_csv(os.path.join(stage1_dir, 'trial_quality.csv'), index=False)
        participant_stats.to_csv(os.path.join(stage1_dir, 'participant_quality.csv'), index=False)
        df_clean.to_csv(os.path.join(stage1_dir, 'stage1_clean.csv'), index=False)
        
        print(f"\n  Saved: {stage1_dir}/")
    except Exception as e:
        print(f"\n  Warning: Could not save files: {e}")
    
    print("="*80)
    
    # Collect statistics for reporting
    excluded_samples = df[~df.set_index(['participant_id', 'question_id']).index.isin(
        accepted_trials.set_index(['participant_id', 'question_id']).index
    )]
    
    # Count good trials lost due to participant exclusion
    good_trials_lost = 0
    good_trials_lost_samples = 0
    if excluded_participants_count > 0:
        for pid in excluded_participants:
            p_trials = trial_stats[trial_stats['participant_id'] == pid]
            good_in_excluded_p = (~p_trials['excluded']).sum() if 'excluded' in p_trials else 0
            good_trials_lost += good_in_excluded_p
            # Count samples from good trials
            if good_in_excluded_p > 0:
                good_trial_ids = p_trials[~p_trials['excluded']][['participant_id', 'question_id']]
                for _, row in good_trial_ids.iterrows():
                    trial_samples = df[(df['participant_id'] == row['participant_id']) & 
                                      (df['question_id'] == row['question_id'])]
                    good_trials_lost_samples += len(trial_samples)
    
    stats = {
        'total_trials': total_trials,
        'total_samples': total_samples,
        'excluded_trials_trial_level': excluded_trials,  # Trials excluded due to quality (before participant exclusion)
        'excluded_trials_trial_level_pct': excluded_trials / total_trials * 100,
        'excluded_trials_participant_level': final_excluded_trials - excluded_trials,  # Additional trials lost due to participant exclusion
        'excluded_trials_participant_level_pct': (final_excluded_trials - excluded_trials) / total_trials * 100,
        'excluded_trials': final_excluded_trials,  # Total excluded (trial-level + participant-level)
        'excluded_trials_pct': final_excluded_trials / total_trials * 100,
        'excluded_samples': len(excluded_samples),
        'excluded_participants': excluded_participants_count,
        'good_trials_lost': good_trials_lost,
        'good_trials_lost_pct': good_trials_lost / total_trials * 100,
        'good_trials_lost_samples': good_trials_lost_samples,
        'kept_trials': final_kept_trials,
        'kept_trials_pct': final_kept_trials / total_trials * 100,
        'kept_samples': len(df_clean)
    }
    
    return df_clean, trial_stats, participant_stats, stats
