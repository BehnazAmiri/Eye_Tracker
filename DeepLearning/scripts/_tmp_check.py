import json
d = json.load(open('outputs/reports/lstm_20260310_144045.json'))
cm = d['metrics']['test']['confusion_matrix']
tn,fp,fn,tp = cm[0][0],cm[0][1],cm[1][0],cm[1][1]
spec = tn/(tn+fp) if tn+fp>0 else 0
rec  = tp/(tp+fn) if tp+fn>0 else 0
bacc = (spec+rec)/2
acc = d['metrics']['test']['accuracy']
mc = d.get('model_config',{})
tc = d.get('training_config',{})
print(f"lr={tc.get('learning_rate')} dropout={mc.get('dropout')} seed=512: BAcc={bacc*100:.1f}%  Acc={acc*100:.1f}%  Spec={spec*100:.0f}%  Rec={rec*100:.0f}%  CM={cm}")
