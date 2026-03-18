"""
Reconstruct stage3_v1 labels CSV from the March 3, 2026 dl_inputs trial files.
These files contain the exact labels from the run that achieved Acc=76.47%.

Run from: d:\MasterThesis\MasterThesis\DeepLearning\
"""
import os, re, pandas as pd

DL_INPUTS_FOLDER = 'outputs/dl_inputs/lstm_20260303_202303_filtered_trials'
OUTPUT_CSV = '../DataMining/results/reports/stage3/stage3_with_labels_v1_backup.csv'

rows = []
for fname in sorted(os.listdir(DL_INPUTS_FOLDER)):
    if not fname.endswith('.csv'):
        continue
    # Parse participant and question from filename: participant_X_question_Y.csv
    m = re.match(r'(participant_\d+)_(question_\d+)\.csv', fname)
    if not m:
        print(f'Skipping unrecognized filename: {fname}')
        continue
    participant_id = m.group(1)
    question_id = m.group(2)

    df = pd.read_csv(os.path.join(DL_INPUTS_FOLDER, fname))
    label = df['randomness_label'].iloc[0]
    rows.append({'participant_id': participant_id, 'question_id': question_id, 'randomness_label': label})

result = pd.DataFrame(rows)
print(f'Reconstructed {len(result)} trials')
print(result['randomness_label'].value_counts())
print()
print('Sample:')
print(result.head(10).to_string())

# Save
result.to_csv(OUTPUT_CSV, index=False)
print(f'\nSaved to: {OUTPUT_CSV}')
