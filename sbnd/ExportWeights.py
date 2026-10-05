"""
Builds the per-event systematic universe weights from the universe spectra
computed in the cafpyana notebook. The notebook gives, for every systematic
source and universe, the selected signal in reco KE and reco cos(theta) bins.
The ratio universe/CV in the bin of an event (looked up with its true KE and
true cos(theta)) becomes that event's weight in that universe; the KE and
cos(theta) ratios are multiplied.

    python3 sbnd/ExportWeights.py --inputs omnifold_export_inputs_sel_start_dedx.pkl

Needs mc_vals_truth_NoNorm.npy from FormatData_SBND.py (same selection).
Writes to --export-dir (default sbnd/exported_weights/ in this repository, so the
shared copy is never overwritten; point SBND_EXPORT_DIR at the new folder to use it):
    <source>_universe_weights.npy   (n_events, n_universes) for every source
    mcstat_universe_weights.npy     Poisson(1) weights for MC statistics
    universe_weights_manifest.json  list of sources with their family
    pot_scale.npy

The input pickle is written by this notebook cell (after the covariance cells):

    import pickle
    keep = ('reco_ke', 'reco_costheta')
    export = dict(
        univ_sig={v: {k: np.asarray(r['sig']) for k, r in univ_results[v].items()} for v in keep},
        cv_sig={v: np.asarray(cv_results[v]['sig_cv']) for v in keep},
        sources={k: dict(family=s['family'], group=str(s['group']), kind=s['kind'])
                 for k, s in ACTIVE_MAIN_SOURCES.items()},
        analysis_bins={v: np.asarray(ANALYSIS_BINS[v]) for v in keep},
        pot_scale=float(pot_scale), final_stage=FINAL_STAGE, n_selected=len(sig_sel_valid))
    with open(f'{OUT_DIR}/omnifold_export_inputs_{FINAL_STAGE}.pkl', 'wb') as f:
        pickle.dump(export, f)
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
from numpy.random import SeedSequence, Generator, PCG64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sbnd_config as C

ap = argparse.ArgumentParser(description='Export per-event universe weights for OmniFold.')
ap.add_argument('--inputs', required=True, help='omnifold_export_inputs_<stage>.pkl from the notebook')
ap.add_argument('--data-dir', default=C.DATA_DIR)
ap.add_argument('--export-dir', default=os.path.join(C.REPO_DIR, 'sbnd/exported_weights/'),
                help='output folder')
ap.add_argument('--n-mcstat', type=int, default=100, help='number of MC-stat universes')
args = ap.parse_args()
out = args.export_dir if args.export_dir.endswith('/') else args.export_dir + '/'
os.makedirs(out, exist_ok=True)

with open(args.inputs, 'rb') as f:
    inp = pickle.load(f)

truth = np.load(args.data_dir + 'mc_vals_truth_NoNorm.npy')
# the notebook bins are in KE, so events are looked up by true KE
true_ke, true_cos = truth[:, C.TRUTH_COL['true_ke']], truth[:, C.TRUTH_COL['true_costheta']]
n_events = len(true_ke)

meta_path = args.data_dir + 'format_meta.json'
if os.path.exists(meta_path):
    meta = json.load(open(meta_path))
    if meta.get('final_stage') != inp['final_stage']:
        sys.exit(f"ERROR: the notebook used {inp['final_stage']} but FormatData used "
                 f"{meta.get('final_stage')}. Rerun FormatData_SBND.py --final-stage {inp['final_stage']}")
if inp['n_selected'] != n_events:
    sys.exit(f"ERROR: the notebook selected {inp['n_selected']} signal events, the OmniFold "
             f"sample has {n_events}. Use the same selection pickle in both.")
print(f"{inp['final_stage']}: {n_events:,} events, {len(inp['sources'])} sources")


def bin_weights(vals, bins, ratio):
    idx = np.clip(np.digitize(vals.clip(bins[0], bins[-1] - 1e-8), bins) - 1, 0, ratio.shape[1] - 1)
    return ratio[:, idx].T.astype(np.float32)


def weights_2d(key):
    r_ke = inp['univ_sig']['reco_ke'][key] / np.clip(inp['cv_sig']['reco_ke'][None, :], 1.0, None)
    r_cos = inp['univ_sig']['reco_costheta'][key] / np.clip(inp['cv_sig']['reco_costheta'][None, :], 1.0, None)
    return (bin_weights(true_ke, inp['analysis_bins']['reco_ke'], r_ke) *
            bin_weights(true_cos, inp['analysis_bins']['reco_costheta'], r_cos))


manifest = {}
for key, s in inp['sources'].items():
    w = weights_2d(key)
    np.save(out + f'{key}_universe_weights.npy', w)
    manifest[key] = dict(s, n_univ=int(w.shape[1]), file=f'{key}_universe_weights.npy')

# combined Flux / GENIE throws, kept only for the cross-check in `screen`
for legacy, ckey in (('bnb', 'flux__Flux'), ('genie', 'genie__GENIE')):
    if ckey in inp['univ_sig']['reco_ke']:
        np.save(out + f'{legacy}_universe_weights.npy', weights_2d(ckey))

kids = SeedSequence(42).spawn(args.n_mcstat)
mcstat = np.stack([Generator(PCG64(k)).poisson(1.0, size=n_events) for k in kids], axis=1)
np.save(out + 'mcstat_universe_weights.npy', mcstat.astype(np.float32))
manifest['mcstat'] = dict(family='mcstat', group='mcstat', kind='poisson',
                          n_univ=args.n_mcstat, file='mcstat_universe_weights.npy')

np.save(out + 'pot_scale.npy', np.array([inp['pot_scale']]))
with open(out + C.MANIFEST, 'w') as f:
    json.dump(manifest, f, indent=1)

fams = {}
for m in manifest.values():
    fams[m['family']] = fams.get(m['family'], 0) + 1
print(f"Wrote {len(manifest)} sources to {out}: " +
      ', '.join(f"{C.FAMILY_LABEL.get(f, f)} {n}" for f, n in fams.items()))
