"""
Data Mining Pipeline - Simple Version
Step by step implementation
"""

import os
import pandas as pd
import numpy as np
from configparser import ConfigParser


def load_config(config_path='config.ini'):
    """Load configuration file."""
    config = ConfigParser()
    if os.path.exists(config_path):
        config.read(config_path)
    return config


def load_gaze_data(outputs_dir, participant_range, question_range):
    """Load gaze data from CSV files."""
    print("\n[1/5] Loading gaze data...")
    
    all_data = []
    loaded_count = 0
    
    for p_id in participant_range:
        participant_folder = os.path.join(outputs_dir, f'participant_{p_id}')
        
        if not os.path.exists(participant_folder):
            continue
        
        for q_id in question_range:
            gaze_file = os.path.join(participant_folder, f'Q{q_id}.csv')
            
            if not os.path.exists(gaze_file):
                continue
            
            try:
                df = pd.read_csv(gaze_file)
                df['participant_id'] = f'participant_{p_id}'
                df['question_id'] = f'question_{q_id}'
                all_data.append(df)
                loaded_count += 1
            except Exception as e:
                print(f"  Warning: Failed to load {gaze_file}: {e}")
    
    if not all_data:
        raise ValueError("No gaze data found!")
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"  Loaded {len(df):,} samples from {loaded_count} files")
    
    return df


def load_answer_data(outputs_dir, participant_range, question_range):
    """
    Load answer data from answers.json files.
    
    Note: time_spent (t_end) is in SECONDS (from time.time() in EyeTracker.py).
    """
    print("\n[2/5] Loading answer data...")
    import json
    
    records = []
    
    for p_id in participant_range:
        participant_folder = os.path.join(outputs_dir, f'participant_{p_id}')
        answers_file = os.path.join(participant_folder, 'answers.json')
        
        if not os.path.exists(answers_file):
            continue
        
        try:
            with open(answers_file, 'r', encoding='utf-8') as f:
                answers = json.load(f)
            
            for answer in answers:
                q_id = answer.get('question_id')
                
                if q_id not in question_range:
                    continue
                
                records.append({
                    'participant_id': f'participant_{p_id}',
                    'question_id': f'question_{q_id}',
                    'chosen_option': answer.get('chosen_option', ''),
                    'answered': 1 if answer.get('chosen_option') else 0,
                    't_end': answer.get('time_spent', None),
                })
        except Exception as e:
            print(f"  Warning: Failed to load {answers_file}: {e}")
    
    answer_df = pd.DataFrame(records)
    print(f"  Loaded {len(answer_df)} answer records")
    
    return answer_df


def load_part_assignments(question_exams_dir, participant_range):
    """Load part assignments (No-Timer, Timer-Correct, Timer-No-Correct)."""
    print("\n[3/5] Loading part assignments...")
    import json
    
    records = []
    
    for filename in os.listdir(question_exams_dir):
        if not filename.endswith('.json'):
            continue
        
        # Extract participant ID: Participant_1.json -> participant_1
        participant_id = filename.replace('.json', '').replace('Participant_', 'participant_')
        
        # Extract numeric ID
        try:
            p_num = int(participant_id.replace('participant_', ''))
            if p_num not in participant_range:
                continue
        except:
            continue
        
        filepath = os.path.join(question_exams_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse structure: {Part1: [...], Part2: [...]}
        for part_name, questions in data.items():
            if isinstance(questions, list):
                for q_data in questions:
                    q_id = q_data.get('question_id')
                    has_correct = q_data.get('has_correct_answer', True)
                    
                    # Map Part1/Part2 to conditions
                    if part_name == 'Part1':
                        part = 'No-Timer'
                    elif part_name == 'Part2':
                        part = 'Timer-Correct' if has_correct else 'Timer-No-Correct'
                    else:
                        part = 'Unknown'
                    
                    records.append({
                        'participant_id': participant_id,
                        'question_id': f'question_{q_id}',
                        'part': part
                    })
    
    part_df = pd.DataFrame(records)
    print(f"  Loaded {len(part_df)} part assignments")
    
    return part_df


def generate_raw_data_report(gaze_df, answer_df, part_df, report_dir):
    """
    Generate comprehensive report of raw data before Stage 1.
    
    Args:
        gaze_df: Raw gaze data
        answer_df: Answer data
        part_df: Part assignments
        report_dir: Output directory
    """
    import os
    
    print("\n" + "="*80)
    print("RAW DATA SUMMARY REPORT")
    print("="*80)
    
    # Overall statistics
    n_participants = gaze_df['participant_id'].nunique()
    n_questions = gaze_df['question_id'].nunique()
    n_samples = len(gaze_df)
    n_trials = len(gaze_df.groupby(['participant_id', 'question_id']))
    
    print(f"\n[OVERALL STATISTICS]")
    print(f"  Total Participants: {n_participants}")
    print(f"  Total Questions: {n_questions}")
    print(f"  Total Trials: {n_trials} ({n_participants} × {n_questions})")
    print(f"  Total Samples: {n_samples:,}")
    print(f"  Average samples per trial: {n_samples/n_trials:.1f}")
    
    # Samples by participant
    participant_summary = gaze_df.groupby('participant_id').agg(
        n_samples=('FPOGS', 'count'),
        n_questions=('question_id', 'nunique')
    ).reset_index()
    participant_summary = participant_summary.sort_values('participant_id')
    
    print(f"\n[SAMPLES BY PARTICIPANT]")
    print(f"  {'Participant':<15} {'Questions':<12} {'Samples':<12} {'Avg/Question':<12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
    for _, row in participant_summary.iterrows():
        avg_per_q = row['n_samples'] / row['n_questions']
        print(f"  {row['participant_id']:<15} {row['n_questions']:<12} {row['n_samples']:<12,} {avg_per_q:<12.1f}")
    
    # Samples by question
    question_summary = gaze_df.groupby('question_id').agg(
        n_samples=('FPOGS', 'count'),
        n_participants=('participant_id', 'nunique')
    ).reset_index()
    question_summary = question_summary.sort_values('question_id')
    
    print(f"\n[SAMPLES BY QUESTION]")
    print(f"  {'Question':<15} {'Participants':<15} {'Samples':<12} {'Avg/Participant':<15}")
    print(f"  {'-'*15} {'-'*15} {'-'*12} {'-'*15}")
    for _, row in question_summary.iterrows():
        avg_per_p = row['n_samples'] / row['n_participants']
        print(f"  {row['question_id']:<15} {row['n_participants']:<15} {row['n_samples']:<12,} {avg_per_p:<15.1f}")
    
    # Trial-level details
    trial_summary = gaze_df.groupby(['participant_id', 'question_id']).agg(
        n_samples=('FPOGS', 'count')
    ).reset_index()
    trial_summary = trial_summary.sort_values(['participant_id', 'question_id'])
    
    # Part distribution
    if 'part' in gaze_df.columns:
        part_summary = gaze_df.groupby('part').agg(
            n_samples=('FPOGS', 'count'),
            n_trials=('participant_id', lambda x: len(x.unique()) * len(gaze_df[gaze_df['part'] == x.name]['question_id'].unique()) if x.name in gaze_df['part'].values else 0)
        ).reset_index()
        
        print(f"\n[EXPERIMENTAL CONDITIONS]")
        print(f"  {'Condition':<20} {'Trials':<10} {'Samples':<12}")
        print(f"  {'-'*20} {'-'*10} {'-'*12}")
        for _, row in part_summary.iterrows():
            # Count trials properly
            n_trials_part = len(gaze_df[gaze_df['part'] == row['part']].groupby(['participant_id', 'question_id']))
            print(f"  {row['part']:<20} {n_trials_part:<10} {row['n_samples']:<12,}")
    
    # Save detailed reports
    stage0_dir = os.path.join(report_dir, 'stage0_raw_data')
    os.makedirs(stage0_dir, exist_ok=True)
    
    # Save participant summary
    participant_summary.to_csv(os.path.join(stage0_dir, 'participants_summary.csv'), index=False)
    
    # Save question summary
    question_summary.to_csv(os.path.join(stage0_dir, 'questions_summary.csv'), index=False)
    
    # Save trial summary
    trial_summary.to_csv(os.path.join(stage0_dir, 'trials_summary.csv'), index=False)
    
    print(f"\n[OUTPUT FILES]")
    print(f"  Saved: {stage0_dir}/")
    print(f"    - participants_summary.csv ({len(participant_summary)} rows)")
    print(f"    - questions_summary.csv ({len(question_summary)} rows)")
    print(f"    - trials_summary.csv ({len(trial_summary)} rows)")
    
    print("="*80)
    
    return participant_summary, question_summary, trial_summary


def run_pipeline(participant_range, question_range, config_path='config.ini'):
    """
    Main pipeline function.
    
    Args:
        participant_range: List of participant IDs [1, 2, 3, ...]
        question_range: List of question IDs [1, 2, 3, ...]
        config_path: Path to config.ini
    """
    
    print("="*80)
    print("DATA MINING PIPELINE - Simple Version")
    print("="*80)
    
    # Get script directory and make all paths relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load config from script directory
    config_path_abs = os.path.join(script_dir, config_path)
    config = load_config(config_path_abs)
    
    # Get paths from config and resolve them relative to script directory
    outputs_dir = config.get('Paths', 'output_dir', fallback='../eye_tracking_app/outputs')
    question_exams_dir = config.get('Paths', 'question_exams_dir', fallback='../eye_tracking_app/inputs/question_exams')
    report_dir = config.get('Paths', 'report_dir', fallback='results/reports')
    
    # Convert relative paths to absolute paths based on script directory
    outputs_dir = os.path.join(script_dir, outputs_dir)
    question_exams_dir = os.path.join(script_dir, question_exams_dir)
    report_dir = os.path.join(script_dir, report_dir)
    
    # Load data
    gaze_df = load_gaze_data(outputs_dir, participant_range, question_range)
    answer_df = load_answer_data(outputs_dir, participant_range, question_range)
    part_df = load_part_assignments(question_exams_dir, participant_range)
    
    # Merge part assignments
    gaze_df = gaze_df.merge(part_df, on=['participant_id', 'question_id'], how='left')
    print(f"\nMerged parts: {len(gaze_df)} samples with part labels")
    
    # Generate raw data report BEFORE Stage 1
    generate_raw_data_report(gaze_df, answer_df, part_df, report_dir)
    
    # Import stages
    from stage1 import run_stage1
    from stage2 import run_stage2
    from stage3 import run_stage3
    from reporting import ReportGenerator, ThresholdComparison
    
    # Initialize reporting
    report_gen = ReportGenerator(report_dir)
    threshold_comp = ThresholdComparison(report_dir)
    
    # Stage 1: Quality Filter
    gaze_clean, trial_stats, participant_stats, stage1_stats = run_stage1(gaze_df, config, report_dir)
    
    # Stage 2: AOI Assignment + ta Detection
    # Note: AOI coordinates are read directly from config inside run_stage2
    # ta (Time-to-Answer) is returned in SECONDS (relative to trial start)
    gaze_with_aoi, ta_df, stage2_stats = run_stage2(gaze_clean, config, report_dir)
    
    # Stage 3: Randomness Labeling
    # Note: t_answer = t_end - ta (both in SECONDS)
    stage3_df, stage3_stats = run_stage3(ta_df, answer_df, config, report_dir)
    
    # Final export
    print("\n" + "="*80)
    print("FINAL EXPORT")
    print("="*80)
    
    try:
        final_dir = os.path.join(report_dir, 'final')
        os.makedirs(final_dir, exist_ok=True)
        
        # Merge randomness labels to sample-level data
        final_df = gaze_with_aoi.merge(
            stage3_df[['participant_id', 'question_id', 'randomness_label', 'randomness_label_num']],
            on=['participant_id', 'question_id'],
            how='left'
        )
        
        # Filter: Keep only trials with valid randomness labels (NOT_RANDOM or RANDOM)
        # NOTE: We keep ALL samples (including is_valid=False) within valid trials
        # DL pipeline is responsible for sample-level quality filtering
        trials_before = final_df[['participant_id', 'question_id']].drop_duplicates().shape[0]
        final_df = final_df[final_df['randomness_label'].isin(['NOT_RANDOM', 'RANDOM'])].copy()
        trials_removed = trials_before - final_df[['participant_id', 'question_id']].drop_duplicates().shape[0]
        
        print(f"\n  Trial-level filtering:")
        print(f"    Removed {trials_removed} trials with INVALID label")
        print(f"    Kept all samples (including is_valid=False) within valid trials")
        
        # Save complete dataset
        final_df.to_csv(os.path.join(final_dir, 'final_dataset.csv'), index=False)
        
        print(f"  Total samples: {len(final_df):,}")
        print(f"  Trials: {final_df[['participant_id', 'question_id']].drop_duplicates().shape[0]}")
        print(f"  Participants: {final_df['participant_id'].nunique()}")
        print(f"\n  Saved: {final_dir}/final_dataset.csv")
        
        # Save per-trial files
        trials_dir = os.path.join(final_dir, 'trials')
        os.makedirs(trials_dir, exist_ok=True)
        
        print(f"\n  Exporting per-trial files...")
        trial_count = 0
        for (participant_id, question_id), trial_data in final_df.groupby(['participant_id', 'question_id']):
            # Extract participant and question numbers
            p_num = participant_id.replace('participant_', '')
            q_num = question_id.replace('question_', '')
            
            # Save trial file: participant_1_question_5.csv
            trial_filename = f"{participant_id}_{question_id}.csv"
            trial_filepath = os.path.join(trials_dir, trial_filename)
            trial_data.to_csv(trial_filepath, index=False)
            trial_count += 1
        
        print(f"  Saved {trial_count} trial files to: {trials_dir}/")
        
    except Exception as e:
        print(f"  Warning: Could not save final dataset: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*80)
    
    # Generate comprehensive HTML report
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*80)
    
    # IMPORTANT: Save this run FIRST before generating comparison tables
    run_index = threshold_comp.add_run(config, stage1_stats, stage2_stats, stage3_stats)
    print(f"\n  [SAVED] Run #{run_index + 1} saved for threshold comparison")
    
    # Get comparison data from ALL runs (including current one)
    comparison_data = threshold_comp.get_comparison_table_data()
    
    # Generate HTML report with complete comparison data
    report_path = report_gen.generate_report(
        config, 
        stage1_stats, 
        stage2_stats, 
        stage3_stats,
        comparison_data
    )
    
    print(f"  [REPORT] HTML Report: {report_path}")
    
    # Open report in default browser
    try:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(report_path))
        print(f"  [OK] Opening report in browser...")
    except Exception as e:
        print(f"  [WARNING] Could not open report: {e}")
    
    print("="*80)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return final_df, stage3_df


if __name__ == "__main__":
    # Full dataset: 30 participants, 15 questions
    participant_range = list(range(1, 31))  # 30 participants
    question_range = list(range(1, 16))  # 15 questions
    
    try:
        final_df, stage3_df = run_pipeline(participant_range, question_range)
        print(f"\n[OK] Pipeline completed!")
        print(f"  Final samples: {len(final_df):,}")
        print(f"  Trials with labels: {len(stage3_df)}")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
