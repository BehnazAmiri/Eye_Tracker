import json, os, glob

files = sorted(glob.glob('DeepLearning/outputs/reports/lstm_*.json'), key=os.path.getmtime, reverse=True)
f = files[0]
print('Latest:', f)
with open(f) as fp:
    d = json.load(fp)

dc = d.get('data_config', {})
mc = d.get('model_config', {})
tc = d.get('training_config', {})
m_test = d.get('metrics', {}).get('test', {})
m_train = d.get('metrics', {}).get('train', {})
dm = d.get('dm_source_config', {})

print('\n=== DATA CONFIG ===')
print('  n_train=%s  n_test=%s  total=%s' % (dc.get('n_train'), dc.get('n_test'), dc.get('total_samples_used')))
print('  class_0=%s  class_1=%s  (0=RANDOM, 1=NOT_RANDOM)' % (dc.get('original_class_0'), dc.get('original_class_1')))
print('  parts=%s' % dc.get('parts_filter'))
print('  seq_len=%s' % dc.get('sequence_length'))

si = dc.get('split_info', {})
print('  train: c0=%s c1=%s  | test: c0=%s c1=%s' % (
    si.get('train_class_0'), si.get('train_class_1'),
    si.get('test_class_0'), si.get('test_class_1')
))

print('\n=== TRAINING CONFIG ===')
print('  lr=%s  batch=%s  patience=%s  dropout=%s' % (
    tc.get('learning_rate'), tc.get('batch_size'),
    tc.get('early_stopping_patience'), mc.get('dropout')
))
print('  epochs_trained=%s  best_epoch=%s' % (
    m_train.get('epochs_trained'), m_train.get('best_epoch')
))
print('  final_train_loss=%s  final_val_loss=%s' % (
    m_train.get('final_loss'), m_train.get('final_val_loss')
))
print('  train_acc=%s' % m_train.get('accuracy'))

print('\n=== TEST METRICS ===')
print('  acc=%.4f  f1=%.4f  auc=%.4f' % (
    m_test.get('accuracy', 0), m_test.get('f1', 0), m_test.get('roc_auc', 0)
))
print('  prec=%.4f  rec=%.4f' % (m_test.get('precision', 0), m_test.get('recall', 0)))
print('  CM=%s' % m_test.get('confusion_matrix'))

print('\n=== DM SOURCE ===')
thresholds = dm.get('thresholds', {})
print('  ta_window=%s  coverage=%s  exclusion=%s  percentile=%s' % (
    thresholds.get('ta_window_ms'), thresholds.get('ta_answer_coverage_threshold'),
    thresholds.get('stage1_participant_exclusion_threshold'),
    thresholds.get('stage3_threshold_percentile')
))
summary = dm.get('output_summary', {})
print('  total_trials=%s  NOT_RANDOM=%s  RANDOM=%s' % (
    summary.get('total_trials'), summary.get('NOT_RANDOM'), summary.get('RANDOM')
))

print('\n=== MODEL CONFIG ===')
for k,v in mc.items():
    print('  %s = %s' % (k, v))
