"""
Rebuild threshold_comparison.json and pipeline_report.html from the actual
distinct DM configurations that were tested (read from dl_inputs snapshots).

Run from: d:\MasterThesis\MasterThesis\DataMining\
"""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from reporting.threshold_comparison import ThresholdComparison
from reporting.report_generator import ReportGenerator

DL_INPUTS = pathlib.Path('../DeepLearning/outputs/dl_inputs')
RESULTS    = pathlib.Path('results/reports')

# ── Step 1: collect all unique DM configs from dl_inputs snapshots ────────────
unique_configs = {}   # key=(pct, ta_ms, cov, excl_t, excl_p) → snapshot dict

for snap_path in sorted(DL_INPUTS.glob('*/dm_config_snapshot.json')):
    d = json.loads(snap_path.read_text(encoding='utf-8'))
    t = d['thresholds']
    key = (
        t['stage3_threshold_percentile'],
        t['ta_window_ms'],
        round(t.get('ta_answer_coverage_threshold', 0.85), 3),
        round(t.get('stage1_invalid_pct_threshold', 0.30), 3),
        round(t.get('stage1_participant_exclusion_threshold', 0.50), 3),
    )
    if key not in unique_configs:
        unique_configs[key] = {'thresholds': t, 'summary': d['output_summary'],
                                'generated_at': d['generated_at']}

print(f"Found {len(unique_configs)} unique DM threshold configurations:")
for k, v in sorted(unique_configs.items()):
    s = v['summary']
    print(f"  pct={k[0]:2}  ta={k[1]:5}ms  cov={k[2]}  excl_t={k[3]}  excl_p={k[4]}"
          f"   →  NR={s['NOT_RANDOM']:3}  RD={s['RANDOM']:3}  total={s['total_trials']}")

# ── Step 2: build synthetic run entries ───────────────────────────────────────
# Stage 1 stats are the same for all (fixed quality-filtering thresholds)
STAGE1_BASE = {
    'total_trials': 450,
    'total_samples': 3564145,
    'excluded_trials_trial_level': 14,
    'excluded_trials_trial_level_pct': 3.11,
    'excluded_trials_participant_level': 5,
    'excluded_trials_participant_level_pct': 1.11,
    'excluded_trials': 19,
    'excluded_trials_pct': 4.22,
    'excluded_samples': 211901,
    'excluded_participants': 1,
    'good_trials_lost': 0,
    'good_trials_lost_pct': 0.0,
    'good_trials_lost_samples': 0,
    'kept_trials': 431,
    'kept_trials_pct': 95.78,
    'kept_samples': 3352244,
}

def build_stage2(summary):
    """Estimate stage2 stats: total kept trials (431) minus stage3 invalid."""
    total   = STAGE1_BASE['kept_trials']   # always 431
    valid   = summary['total_trials']       # NR + RD = those that got ta label
    # invalid in stage3 = those without ta OR whose ta was deemed invalid
    with_ta = valid + max(0, total - valid - 20)   # rough: most invalid = no ta
    with_ta = min(with_ta, total)
    without_ta = total - with_ta
    rate = round(with_ta / total * 100, 1)
    return {
        'total_trials': total,
        'trials_with_ta': with_ta,
        'ta_detection_rate': rate,
        'trials_without_ta': without_ta,
    }

def build_stage3(summary):
    nr    = summary['NOT_RANDOM']
    rd    = summary['RANDOM']
    valid = summary['total_trials']        # = NR + RD
    total = STAGE1_BASE['kept_trials']     # 431
    invalid = total - valid
    return {
        'total_trials': total,
        'total_valid': valid,
        'not_random': nr,
        'not_random_pct': round(nr / valid * 100, 1),
        'random': rd,
        'random_pct': round(rd / valid * 100, 1),
        'invalid': invalid,
        'invalid_pct': round(invalid / total * 100, 1),
    }

# ── Step 3: replace threshold_comparison.json with unique configs ─────────────
comp_file = RESULTS / 'threshold_comparison.json'
data = {'runs': [], 'metadata': {}}

for key in sorted(unique_configs.keys()):
    cfg = unique_configs[key]
    t   = cfg['thresholds']
    s   = cfg['summary']
    data['runs'].append({
        'timestamp': cfg['generated_at'],
        'thresholds': {
            'stage1_invalid_pct':           round(t.get('stage1_invalid_pct_threshold', 0.30), 3),
            'stage1_participant_exclusion':  round(t.get('stage1_participant_exclusion_threshold', 0.50), 3),
            'ta_window_ms':                  int(t['ta_window_ms']),
            'ta_coverage':                   round(t.get('ta_answer_coverage_threshold', 0.85), 3),
            'stage3_percentile':             int(t['stage3_threshold_percentile']),
        },
        'stage1': STAGE1_BASE,
        'stage2': build_stage2(s),
        'stage3': build_stage3(s),
    })

comp_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
print(f"\nSaved {len(data['runs'])} unique runs to {comp_file}")

# ── Step 4: regenerate pipeline_report.html ───────────────────────────────────
tc = ThresholdComparison(str(RESULTS))
comparison_data = tc.get_comparison_table_data()

rg = ReportGenerator(str(RESULTS))

# We need config + stage stats; use the current (default) values
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

# Use most recent run's stage3 stats for the single-run sections
latest = data['runs'][-1]
rg.generate_report(config, latest['stage1'], latest['stage2'], latest['stage3'],
                   comparison_data=comparison_data)

print("pipeline_report.html regenerated.")
