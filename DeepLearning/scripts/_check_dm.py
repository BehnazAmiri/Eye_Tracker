import pandas as pd
df = pd.read_csv('outputs/reports/experiment_report.csv')
dm_runs = df[df['DM_Pct'].notna()].copy()
print('DM runs total:', len(dm_runs))

combos = dm_runs.groupby(['DM_Pct','DM_ta_ms']).agg(
    Runs=('BAcc','count'), BestBAcc=('BAcc','max'), BestAcc=('Acc','max')
).reset_index()
print(combos.to_string())
print()

print('=== Best per model per DM config ===')
for _, combo in combos.iterrows():
    pct = combo['DM_Pct']; ta = combo['DM_ta_ms']
    sub = dm_runs[(dm_runs['DM_Pct']==pct) & (dm_runs['DM_ta_ms']==ta)]
    best = sub.sort_values('BAcc', ascending=False).drop_duplicates('Model')
    print(f"DM_Pct={pct} ta_ms={ta} ({int(combo['Runs'])} runs):")
    print(best[['Model','BAcc','Acc','Recall','Spec','F1','AUC','Hidden','Layers','Dropout','LR','Batch','Seed']].to_string())
    print()
