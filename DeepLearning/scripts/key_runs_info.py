import json
files = [
    ('outputs/reports/lstm_20260215_224242.json', 'Best Accuracy (76%)'),
    ('outputs/reports/lstm_20260216_231435.json', 'Best BAcc (72.98%)'),
    ('outputs/reports/lstm_20260306_181826.json', 'Current Best seed=512'),
]
for f, label in files:
    d = json.load(open(f))
    m = d['metrics']['test']
    cm = m['confusion_matrix']
    tn,fp_,fn,tp = cm[0][0],cm[0][1],cm[1][0],cm[1][1]
    spec = tn/(tn+fp_)
    rec  = tp/(tp+fn)
    acc  = m['accuracy']
    bacc = (spec+rec)/2
    tr   = d['training_config']
    mc   = d['model_config']
    dc   = d['data_config']
    print(f"--- {label} ---")
    print(f"  Acc={acc*100:.2f}%  BAcc={bacc*100:.2f}%  Recall={rec*100:.2f}%  Spec={spec*100:.2f}%")
    print(f"  CM: TN={tn} FP={fp_} FN={fn} TP={tp}")
    print(f"  LR={tr['learning_rate']}  batch={tr['batch_size']}  dropout={mc['dropout']}  hidden={mc['hidden_size']}  layers={mc['num_layers']}")
    print(f"  seed={dc.get('random_seed')}  trials={dc.get('total_samples_used')}  NR={dc.get('original_class_0')}  RD={dc.get('original_class_1')}")
    dm = d.get('dm_source_config', {})
    if dm:
        thr = dm.get('thresholds', {})
        print(f"  DM: pct={thr.get('stage3_threshold_percentile')}  ta_ms={thr.get('ta_window_ms')}  cov={thr.get('ta_answer_coverage_threshold')}")
    print()
