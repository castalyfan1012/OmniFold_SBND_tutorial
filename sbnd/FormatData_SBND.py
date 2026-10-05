"""
Prepares the OmniFold inputs from the selection pickle made by the cafpyana notebook.

The sample is the true signal events that pass the final selection cut. For each
event we keep three reco and three truth variables (KE, cos(theta), p), save them
both raw and standardised, and compute the selection efficiency in the analysis
binning.

    python3 sbnd/FormatData_SBND.py
    python3 sbnd/FormatData_SBND.py --sel-file <pickle> --final-stage sel_start_dedx

Outputs in --data-dir ($SBND_DATA_DIR):
    mc_vals_reco.npy, mc_vals_truth.npy        standardised inputs for the networks
    mc_vals_truth_NoNorm.npy                   raw truth values, used for all histograms
    mc_pass_reco.npy, mc_pass_truth.npy        all True (every event is selected signal)
    mc_weights_reco.npy, mc_weights_truth.npy  POT scale factor per event
    efficiency_<var>.npy                       selection efficiency per bin
    format_meta.json, mc_event_index.pkl       what was selected, and in which order

If the number of events changes, the universe weights must be exported again and
every OmniFold training redone.
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sbnd_config as C

ap = argparse.ArgumentParser(description='Format the selected nueCC events for OmniFold.')
ap.add_argument('--sel-file', default=C.SEL_FILE, help='selection pickle (default $SBND_SEL_FILE)')
ap.add_argument('--data-dir', default=C.DATA_DIR, help='output folder (default $SBND_DATA_DIR)')
ap.add_argument('--export-dir', default=C.EXPORT_DIR,
                help='only used to read pot_scale.npy if the pickle has no pot_scale column')
ap.add_argument('--final-stage', default=C.FINAL_STAGE,
                help='selection column that defines the sample (default $SBND_FINAL_STAGE)')
args = ap.parse_args()

SEL_FILE    = args.sel_file
OUTPUT_DIR  = args.data_dir if args.data_dir.endswith('/') else args.data_dir + '/'
EXPORT_DIR  = args.export_dir if args.export_dir.endswith('/') else args.export_dir + '/'
FINAL_STAGE = args.final_stage
RECO_VARS   = ['reco_ke', 'reco_costheta', 'reco_p']
TRUTH_VARS  = ['true_ke', 'true_costheta', 'true_p']
assert all(C.TRUTH_COL[v] == i for i, v in enumerate(TRUTH_VARS))
EFF_THRESHOLD = 0.10   # bins below this efficiency are flagged

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading {SEL_FILE}")
sel_topo = pd.read_pickle(SEL_FILE)
if sel_topo.index.duplicated().any():
    sel_topo = sel_topo[~sel_topo.index.duplicated(keep='first')]
if 'pot_scale' in sel_topo:
    pot_scale = float(sel_topo['pot_scale'].iloc[0])
else:
    pot_scale = float(np.load(EXPORT_DIR + 'pot_scale.npy')[0])
print(f"  {len(sel_topo):,} rows, POT scale {pot_scale:.4f}")

if FINAL_STAGE not in sel_topo.columns:
    stages = [c for c in sel_topo.columns if c.startswith('sel_')]
    sys.exit(f"ERROR: '{FINAL_STAGE}' is not a column of {SEL_FILE}. Available: {stages}")

selected_signal = sel_topo[sel_topo['is_sig'] & sel_topo[FINAL_STAGE]].copy()
print(f"\nSelected signal events (is_sig & {FINAL_STAGE}): {len(selected_signal):,}")

reco_raw = selected_signal[RECO_VARS].values.astype(np.float32)
valid = ~np.isnan(reco_raw).any(axis=1)
if (~valid).sum() > 0:
    print(f"  dropping {(~valid).sum()} events without a reconstructed shower")
    selected_signal = selected_signal[valid]
    reco_raw = reco_raw[valid]

truth_raw = selected_signal[TRUTH_VARS].values.astype(np.float32)
n = len(selected_signal)
print(f"  Final N (reco & truth): {n:,}")

for name, arr in [('reco', reco_raw), ('truth', truth_raw)]:
    nans = np.isnan(arr).sum()
    if nans > 0:
        print(f"  WARNING: {nans} NaNs in {name}, replaced by the column mean")
        col_means = np.nanmean(arr, axis=0)
        inds = np.where(np.isnan(arr))
        arr[inds] = col_means[inds[1]]

pass_flags = np.ones(n, dtype=bool)
weights = np.full(n, pot_scale, dtype=np.float32)
reco_norm = StandardScaler().fit_transform(reco_raw).astype(np.float32)
truth_norm = StandardScaler().fit_transform(truth_raw).astype(np.float32)

np.save(OUTPUT_DIR + 'mc_vals_reco.npy', reco_norm)
np.save(OUTPUT_DIR + 'mc_vals_truth.npy', truth_norm)
np.save(OUTPUT_DIR + 'mc_vals_truth_NoNorm.npy', truth_raw)
np.save(OUTPUT_DIR + 'mc_pass_reco.npy', pass_flags)
np.save(OUTPUT_DIR + 'mc_pass_truth.npy', pass_flags)
np.save(OUTPUT_DIR + 'mc_weights_reco.npy', weights)
np.save(OUTPUT_DIR + 'mc_weights_truth.npy', weights)
selected_signal.index.to_frame(index=False).to_pickle(OUTPUT_DIR + 'mc_event_index.pkl')
with open(OUTPUT_DIR + 'format_meta.json', 'w') as fh:
    json.dump(dict(n_events=int(n), final_stage=FINAL_STAGE, pot_scale=pot_scale,
                   truth_cols=TRUTH_VARS, reco_cols=RECO_VARS, sel_file=SEL_FILE,
                   binning={k: v.tolist() for k, v in C.BINNING.items()},
                   created=datetime.now().isoformat(timespec='seconds')), fh, indent=1)
print(f"\nSaved {n:,} events to {OUTPUT_DIR}")

print("\nVariable ranges:")
for i, v in enumerate(RECO_VARS):
    print(f"  {v:15s} {np.nanmin(reco_raw[:, i]):9.2f} - {np.nanmax(reco_raw[:, i]):9.2f}")
for i, v in enumerate(TRUTH_VARS):
    print(f"  {v:15s} {np.nanmin(truth_raw[:, i]):9.2f} - {np.nanmax(truth_raw[:, i]):9.2f}")
ev = truth_raw[:, C.TRUTH_COL[C.EVAR]]
lo, hi = C.BINNING[C.EVAR][0], C.BINNING[C.EVAR][-1]
print(f"  {C.EVAR} below {lo:.0f}: {(ev < lo).sum():,} events (outside the measurement)")
print(f"  {C.EVAR} above {hi:.0f}: {(ev >= hi).sum():,} events (go into the last bin)")

# Efficiency = selected signal / all signal, per bin of the true variables
all_signal = sel_topo[sel_topo['is_sig']]
print(f"\nEfficiency ({n:,} selected out of {len(all_signal):,} signal events)")
for var_name in C.VARS:
    bins = C.BINNING[var_name]
    gen_vals = all_signal[var_name].values.astype(np.float64)
    gen_vals = gen_vals[~np.isnan(gen_vals)]
    N_gen = C.hist(var_name, gen_vals)
    N_sel = C.hist(var_name, selected_signal[var_name].values.astype(np.float64))
    eff = np.where(N_gen > 0, N_sel / np.maximum(N_gen, 1), 0.0)
    reliable = eff >= EFF_THRESHOLD
    np.save(OUTPUT_DIR + f'efficiency_{var_name}.npy', eff)
    np.save(OUTPUT_DIR + f'efficiency_mask_{var_name}.npy', reliable)
    np.savez(OUTPUT_DIR + f'efficiency_counts_{var_name}.npz', N_gen=N_gen, N_sel=N_sel, bins=bins)

    labels = C.bin_labels(var_name)
    print(f"\n  {var_name}")
    print(f"  {'bin':>16s} {'all':>7s} {'selected':>9s} {'eff':>7s}")
    for i in range(len(bins) - 1):
        flag = '' if reliable[i] else f'  (below {EFF_THRESHOLD:.0%})'
        print(f"  {labels[i]:>16s} {N_gen[i]:7.0f} {N_sel[i]:9.0f} {eff[i]:7.3f}{flag}")

print("\nNext: python3 sbnd/RunStudies.py list-sources")
