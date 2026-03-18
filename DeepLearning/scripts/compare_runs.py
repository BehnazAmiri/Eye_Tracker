import json, os

for ts, label in [('lstm_20260305_114940', 'GOOD-n51'), ('lstm_20260305_112319', 'BAD-n54'), ('lstm_20260305_161546', 'LATEST-n54')]:
    fpath = f'DeepLearning/outputs/reports/{ts}.json'
    if not os.path.exists(fpath):
        print(f"Missing: {fpath}")
        continue
    with open(fpath) as f:
        d = json.load(f)
    tc = d.get('training_config', {})
    dm = d.get('dm_source_config', {})
    dc = d.get('data_config', {})
    mc = d.get('model_config', {})
    print(f"\n=== {ts} ({label}) ===")
    print(f"  dm_source_config: {dm}")
    print(f"  model seq_len: {mc.get('sequence_length','?')}")
    print(f"  n_train={dc.get('n_train','?')}  n_test={dc.get('n_test','?')}  total={dc.get('total_samples_used','?')}")
    print(f"  class_0={dc.get('original_class_0','?')}  class_1={dc.get('original_class_1','?')}")
    # look for any data path info
    for key,val in tc.items():
        if 'path' in key.lower() or 'dir' in key.lower() or 'input' in key.lower():
            print(f"  tc.{key} = {val}")
    # config_snapshot
    cs = d.get('config_snapshot', {})
    for section, vals in cs.items():
        if isinstance(vals, dict):
            for k,v in vals.items():
                if 'path' in k.lower() or 'dir' in k.lower() or 'input' in k.lower():
                    print(f"  cfg[{section}][{k}] = {v}")
