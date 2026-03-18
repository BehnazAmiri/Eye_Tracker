import json, glob, os

files = glob.glob('outputs/reports/lstm_2026030*.json')
results = []
for f in files:
    try:
        d = json.load(open(f))
        cm = d['metrics']['test']['confusion_matrix']
        tn,fp,fn,tp = cm[0][0],cm[0][1],cm[1][0],cm[1][1]
        spec = tn/(tn+fp) if tn+fp>0 else 0
        rec  = tp/(tp+fn) if tp+fn>0 else 0
        bacc = (spec+rec)/2
        acc  = d['metrics']['test']['accuracy']
        mc = d.get('model_config',{})
        tc = d.get('training_config',{})
        dc = d.get('data_config',{})
        results.append({
            'name': os.path.basename(f).replace('.json',''),
            'bacc': bacc, 'acc': acc, 'spec': spec, 'rec': rec, 'cm': cm,
            'seed': dc.get('random_seed'),
            'lr': tc.get('learning_rate'),
            'dropout': mc.get('dropout'),
            'hidden': mc.get('hidden_size'),
            'layers': mc.get('num_layers'),
        })
    except:
        pass

results.sort(key=lambda x: x['bacc'], reverse=True)
print('Name                         BAcc   Acc   Spec  Rec  CM                       seed  lr       dropout  h    L')
print('-'*115)
for r in results[:25]:
    print(f"{r['name']}  {r['bacc']*100:.1f}%  {r['acc']*100:.1f}%  {r['spec']*100:.0f}%  {r['rec']*100:.0f}%  {str(r['cm']):<28} {str(r['seed']):<5} {str(r['lr']):<8} {str(r['dropout']):<8} {str(r['hidden']):<5} {str(r['layers'])}")
