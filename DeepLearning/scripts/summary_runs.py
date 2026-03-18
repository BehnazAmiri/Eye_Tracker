import json

runs = [
    ('lstm_20260310_145527', 'batch16 lr=0.0004 drop=0.15 seed=512 (TODAY)'),
    ('lstm_20260310_144045', 'batch8  lr=0.0004 drop=0.15 seed=512 (TODAY-2)'),
    ('lstm_20260306_181826', 'batch8  lr=0.0005 drop=0.12 seed=512 (best repro 68.9%)'),
    ('lstm_20260305_114940', 'batch16 lr=0.0004 drop=0.15 seed=None (76.5% Acc orig)'),
    ('lstm_20260303_223021', 'batch16 lr=0.0004 drop=0.15 seed=None (72.8% BAcc orig)'),
]

print(f"{'BAcc':>7} {'Acc':>7} {'Spec':>6} {'Rec':>6}  CM                  Config")
print('-'*110)
for name, label in runs:
    try:
        d = json.load(open(f'outputs/reports/{name}.json'))
        cm = d['metrics']['test']['confusion_matrix']
        tn,fp,fn,tp = cm[0][0],cm[0][1],cm[1][0],cm[1][1]
        spec = tn/(tn+fp) if tn+fp>0 else 0
        rec  = tp/(tp+fn) if tp+fn>0 else 0
        bacc = (spec+rec)/2
        acc  = d['metrics']['test']['accuracy']
        print(f'{bacc*100:>6.1f}% {acc*100:>6.1f}% {spec*100:>5.0f}% {rec*100:>5.0f}%  {str(cm):<26} {label}')
    except Exception as e:
        print(f'ERROR {name}: {e}')
