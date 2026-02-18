"""
Stage 3: Randomness Labeling using t_answer thresholds
"""

import pandas as pd
import os


def run_stage3(ta_df, answer_df, config, report_dir='results/reports'):
    """
    Stage 3: Classify trials as RANDOM or NOT_RANDOM.
    
    Formula: t_answer = t_end - ta (both in SECONDS)
      - t_end: Total trial duration from answers.json (seconds)
      - ta: Time-to-Answer from Stage 2 (seconds, relative to trial start)
    
    Classification:
      1) Unanswered (answered=0) -> INVALID
      2) Timer-No-Correct -> ALL RANDOM
      3) No-Timer & Timer-Correct -> Use P25 threshold per part
         - t_answer < P25 -> RANDOM
         - t_answer >= P25 -> NOT_RANDOM
    """
    
    print("\n" + "="*80)
    print("STAGE 3: RANDOMNESS LABELING")
    print("="*80)
    
    # Merge ta with answer data
    stage3_df = ta_df.copy()
    
    if answer_df is not None and not answer_df.empty:
        stage3_df = stage3_df.merge(
            answer_df[['participant_id', 'question_id', 'answered', 't_end']],
            on=['participant_id', 'question_id'],
            how='left'
        )
    
    # Initialize labels
    stage3_df['randomness_label'] = 'NOT_RANDOM'
    stage3_df['randomness_label_num'] = 0
    
    # STEP 0: Mark trials without ta as INVALID (cannot compute t_answer)
    if 'ta' in stage3_df.columns:
        no_ta = stage3_df['ta'].isna()
        stage3_df.loc[no_ta, 'randomness_label'] = 'INVALID'
        stage3_df.loc[no_ta, 'randomness_label_num'] = -1
        print(f"  INVALID: {no_ta.sum()} trials without ta")
    
    # STEP 1: Mark unanswered as INVALID
    if 'answered' in stage3_df.columns:
        unanswered = (stage3_df['answered'] == 0) & (stage3_df['randomness_label'] != 'INVALID')
        stage3_df.loc[unanswered, 'randomness_label'] = 'INVALID'
        stage3_df.loc[unanswered, 'randomness_label_num'] = -1
        if unanswered.sum() > 0:
            print(f"  INVALID: {unanswered.sum()} additional unanswered trials")
    
    # STEP 2: Timer-No-Correct = ALL RANDOM
    if 'part' in stage3_df.columns:
        timer_no_correct = (stage3_df['part'] == 'Timer-No-Correct') & (stage3_df['randomness_label'] != 'INVALID')
        stage3_df.loc[timer_no_correct, 'randomness_label'] = 'RANDOM'
        stage3_df.loc[timer_no_correct, 'randomness_label_num'] = 1
        print(f"  Timer-No-Correct: {timer_no_correct.sum()} trials marked RANDOM")
    
    # STEP 3: Calculate t_answer and apply P25 threshold
    percentile = config.getfloat('Analysis', 'stage3_threshold_percentile', fallback=25.0)
    
    if 't_end' in stage3_df.columns and 'ta' in stage3_df.columns:
        stage3_df['t_answer'] = stage3_df['t_end'] - stage3_df['ta']
        print(f"\n  Calculated t_answer = t_end - ta")
        
        # === CRITICAL VALIDATION: Time Unit Consistency ===
        print(f"\n  [TIME UNIT VALIDATION]")
        
        # Check 1: Negative t_answer (CRITICAL ERROR)
        negative_ta = (stage3_df['t_answer'] < 0).sum()
        if negative_ta > 0:
            print(f"  ⚠️  WARNING: {negative_ta} trials have NEGATIVE t_answer!")
            print(f"      This indicates ta > t_end (impossible if units are correct)")
            sample = stage3_df[stage3_df['t_answer'] < 0][['participant_id', 'question_id', 't_end', 'ta', 't_answer']].head(3)
            print(f"\n      Sample cases:")
            for _, row in sample.iterrows():
                print(f"        {row['participant_id']}, {row['question_id']}: t_end={row['t_end']:.2f}, ta={row['ta']:.2f}, t_answer={row['t_answer']:.2f}")
            
            # Mark negative t_answer as INVALID (data corruption or unit mismatch)
            negative_mask = stage3_df['t_answer'] < 0
            stage3_df.loc[negative_mask, 'randomness_label'] = 'INVALID'
            stage3_df.loc[negative_mask, 'randomness_label_num'] = -1
            print(f"      → Marked {negative_ta} trials as INVALID")
        
        # Check 2: Unrealistic t_end values (likely unit mismatch)
        valid_t_end = stage3_df['t_end'].dropna()
        if len(valid_t_end) > 0:
            t_end_min, t_end_max = valid_t_end.min(), valid_t_end.max()
            t_end_median = valid_t_end.median()
            print(f"  t_end range: [{t_end_min:.2f}, {t_end_max:.2f}]s, median={t_end_median:.2f}s")
            
            # If t_end looks like milliseconds (typical trial > 1000ms = 1s)
            if t_end_median > 1000:
                print(f"  ⚠️  WARNING: t_end median > 1000 suggests MILLISECONDS, not seconds!")
                print(f"      Expected: 10-180s per trial. Got median: {t_end_median:.1f}")
            elif t_end_median < 5:
                print(f"  ⚠️  WARNING: t_end median < 5s seems too fast for typical trials")
            else:
                print(f"  [OK] t_end values appear reasonable (seconds)")
        
        # Check 3: Unrealistic ta values
        valid_ta = stage3_df['ta'].dropna()
        if len(valid_ta) > 0:
            ta_min, ta_max = valid_ta.min(), valid_ta.max()
            ta_median = valid_ta.median()
            print(f"  ta range: [{ta_min:.2f}, {ta_max:.2f}]s, median={ta_median:.2f}s")
            
            # ta should be < t_end always
            if ta_median > valid_t_end.median() * 0.9:
                print(f"  ⚠️  WARNING: ta median suspiciously close to t_end median")
            elif ta_max > 300:  # 5 minutes
                print(f"  ⚠️  WARNING: Some ta values > 5 minutes (unlikely for answer area fixation)")
            else:
                print(f"  [OK] ta values appear reasonable (seconds)")
        
        # Check 4: t_answer distribution sanity
        valid_t_answer = stage3_df['t_answer'].dropna()
        if len(valid_t_answer) > 0:
            ta_min, ta_max = valid_t_answer.min(), valid_t_answer.max()
            ta_median = valid_t_answer.median()
            print(f"  t_answer range: [{ta_min:.2f}, {ta_max:.2f}]s, median={ta_median:.2f}s")
            
            if ta_median < 0:
                print(f"  ❌ ERROR: t_answer median is NEGATIVE - unit mismatch or logic error!")
            elif ta_median < 1:
                print(f"  ⚠️  WARNING: t_answer median < 1s seems very fast")
            else:
                print(f"  [OK] t_answer values appear reasonable (seconds)")
        
        print(f"  " + "="*60)
        
        # Check t_end availability
        n_with_t_end = stage3_df['t_end'].notna().sum()
        print(f"\n  Trials with t_end: {n_with_t_end}/{len(stage3_df)} ({n_with_t_end/len(stage3_df):.1%})")
        
        if 'part' in stage3_df.columns:
            print(f"\n  Per-part P{percentile:.0f} thresholds:")
            
            for part in ['No-Timer', 'Timer-Correct']:
                # Filter to this part
                part_mask = (
                    (stage3_df['part'] == part) &
                    (stage3_df['randomness_label'] != 'INVALID') &
                    (stage3_df['ta'].notna())
                )
                part_df = stage3_df[part_mask]
                
                # Get valid t_answer
                valid_t_answer = part_df['t_answer'].dropna()
                
                if len(valid_t_answer) > 0:
                    threshold = valid_t_answer.quantile(percentile / 100.0)
                    
                    print(f"\n    {part}:")
                    print(f"      Trials with ta: {len(part_df)}")
                    print(f"      Trials with t_answer: {len(valid_t_answer)}")
                    print(f"      P{percentile:.0f} = {threshold:.2f}s")
                    
                    # Apply threshold
                    random_mask = (
                        (stage3_df['part'] == part) &
                        (stage3_df['t_answer'] < threshold) &
                        (stage3_df['randomness_label'] != 'INVALID') &
                        (stage3_df['ta'].notna())
                    )
                    stage3_df.loc[random_mask, 'randomness_label'] = 'RANDOM'
                    stage3_df.loc[random_mask, 'randomness_label_num'] = 1
                    
                    n_random = random_mask.sum()
                    print(f"      Classified RANDOM: {n_random}/{len(valid_t_answer)} ({n_random/len(valid_t_answer)*100:.1f}%)")
                else:
                    print(f"\n    {part}: No valid t_answer values")
    
    # Final distribution
    print(f"\n  Final Distribution:")
    label_counts = stage3_df['randomness_label'].value_counts()
    total_trials = len(stage3_df)
    total_valid = label_counts.get('NOT_RANDOM', 0) + label_counts.get('RANDOM', 0)
    
    for label in ['NOT_RANDOM', 'RANDOM', 'INVALID']:
        count = label_counts.get(label, 0)
        pct_of_total = count/total_trials*100
        if label != 'INVALID':
            pct_of_valid = count/total_valid*100 if total_valid > 0 else 0
            print(f"    {label}: {count} ({pct_of_total:.1f}% of all trials, {pct_of_valid:.1f}% of valid trials)")
        else:
            print(f"    {label}: {count} ({pct_of_total:.1f}% of all trials)")
    
    # Save
    try:
        stage3_dir = os.path.join(report_dir, 'stage3')
        os.makedirs(stage3_dir, exist_ok=True)
        
        stage3_df.to_csv(os.path.join(stage3_dir, 'stage3_with_labels.csv'), index=False)
        
        print(f"\n  Saved: {stage3_dir}/")
    except Exception as e:
        print(f"\n  Warning: Could not save files: {e}")
    
    print("="*80)
    
    # Collect statistics for reporting
    total_trials = len(stage3_df)
    not_random = (stage3_df['randomness_label'] == 'NOT_RANDOM').sum()
    random = (stage3_df['randomness_label'] == 'RANDOM').sum()
    invalid = (stage3_df['randomness_label'] == 'INVALID').sum()
    total_valid = not_random + random
    
    # Per-part breakdown
    by_part = {}
    if 'part' in stage3_df.columns:
        for part_name in ['No-Timer', 'Timer-Correct', 'Timer-No-Correct']:
            part_df = stage3_df[stage3_df['part'] == part_name]
            if len(part_df) > 0:
                by_part[part_name] = {
                    'total': len(part_df),
                    'not_random': (part_df['randomness_label'] == 'NOT_RANDOM').sum(),
                    'random': (part_df['randomness_label'] == 'RANDOM').sum(),
                    'invalid': (part_df['randomness_label'] == 'INVALID').sum()
                }
    
    stats = {
        'total_trials': total_trials,
        'total_valid': total_valid,
        'not_random': not_random,
        'not_random_pct': not_random / total_valid * 100 if total_valid > 0 else 0,
        'random': random,
        'random_pct': random / total_valid * 100 if total_valid > 0 else 0,
        'invalid': invalid,
        'invalid_pct': invalid / total_trials * 100 if total_trials > 0 else 0,
        'by_part': by_part
    }
    
    return stage3_df, stats
