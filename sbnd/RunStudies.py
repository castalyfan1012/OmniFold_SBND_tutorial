"""
OmniFold studies for the SBND nueCC analysis. Run from the repo root:

    python3 sbnd/RunStudies.py <action> [options]      (add -h after an action for its options)

    list-sources    systematic sources in the exported weights, and how many are trained
    screen          size of every source without any training; picks the ones to train
    check-closure   checks the closure test (nominal MC unfolded onto itself)
    make-fakedata   makes fake data by tilting the MC truth (or from one universe)
    run-syst        unfolds every universe of the chosen sources (one training each)
    run-ml-unc      repeats the fake-data unfolding with different random seeds
    status          progress of run-syst and run-ml-unc

Examples:
    python3 sbnd/RunStudies.py screen --thresh 0.005
    python3 sbnd/RunStudies.py make-fakedata --mode tilt --var true_p --alpha 0.3
    python3 sbnd/RunStudies.py run-syst --source genie__MaCCRES --start 0 --end 10
    python3 sbnd/RunStudies.py run-syst --source screened --worker 0 --n-workers 3
    python3 sbnd/RunStudies.py run-ml-unc --n-replicas 50 --var true_p --alpha 0.3
"""

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import glob
import json
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sbnd_config as C
from sbnd_config import BINNING, XLABEL, chi2_pvalue, make_fdt_tag, tilt_label

SCREEN_FILE = os.path.join(C.COV_DIR, 'screen_sources.json')

# CLI
parser = argparse.ArgumentParser(description='SBND OmniFold studies (see the top of this file).')
sub = parser.add_subparsers(dest='action', required=True)


def _paths(p, export=True):
    p.add_argument('--data-dir', default=C.DATA_DIR, help='FormatData output (default $SBND_DATA_DIR)')
    if export:
        p.add_argument('--export-dir', default=C.EXPORT_DIR,
                       help='exported universe weights (default $SBND_EXPORT_DIR)')


def _norm(p):
    p.add_argument('--keep-norm', dest='keep_norm', action='store_true',
                   help='universes keep their normalisation (default, see KEEP_NORM in sbnd_config.py)')
    p.add_argument('--shape-only', dest='keep_norm', action='store_false',
                   help='rescale every universe to the nominal total')
    p.set_defaults(keep_norm=C.KEEP_NORM)


p_ls = sub.add_parser('list-sources', help='list the systematic sources')
_paths(p_ls)
p_ls.add_argument('--weights-base', default=C.WEIGHTS_BASE, help='where trained universes are')

p_sc = sub.add_parser('screen', help='size of each source without training')
_paths(p_sc)
_norm(p_sc)
p_sc.add_argument('--thresh', type=float, default=0.005,
                  help='train a source if its shape-only uncertainty is above this fraction '
                       'in any bin (default 0.005 = 0.5%%)')

p_cl = sub.add_parser('check-closure', help='check the closure test')
p_cl.add_argument('--closure-dir', default=None, help='default: $SBND_RUNS_DIR/weights_sbnd_closure/')
p_cl.add_argument('--data-dir', default=C.DATA_DIR)
p_cl.add_argument('--plot-dir', default='sbnd/plots_validation')
p_cl.add_argument('--pval-thresh', type=float, default=0.05, help='pass if p-value is above this')
p_cl.add_argument('--var', choices=C.VARS + ['both'], default='both', help='variable(s) to check')
p_cl.add_argument('--show-trials', action='store_true')

p_fd = sub.add_parser('make-fakedata', help='make fake data')
p_fd.add_argument('--mode', choices=['tilt', 'universe'], default='tilt',
                  help='tilt: reweight the truth by 1 + alpha*z; universe: use one systematic universe')
p_fd.add_argument('--var', choices=C.TILT_CHOICES, default=C.EVAR,
                  help='variable to tilt (both = energy and angle together)')
p_fd.add_argument('--alpha', type=float, default=0.3, help='tilt strength')
p_fd.add_argument('--source', default='flux__horncurrent', help='universe mode: which source')
p_fd.add_argument('--universe-idx', type=int, default=0, help='universe mode: which universe')
p_fd.add_argument('--universe-file', type=str, default=None)
p_fd.add_argument('--plot-dir', default='sbnd/plots_validation')
_paths(p_fd)

p_sy = sub.add_parser('run-syst', help='unfold systematic universes')
p_sy.add_argument('--source', required=True,
                  help="source name, family, group (all, fds, xsec, syst), a comma list, "
                       "or 'screened' for the sources chosen by `screen`")
p_sy.add_argument('--only-screened', action='store_true',
                  help='with a family or group: skip the sources `screen` did not choose')
p_sy.add_argument('--start', type=int, default=0, help='first universe index')
p_sy.add_argument('--end', type=int, default=None, help='stop before this universe (default: all)')
p_sy.add_argument('--niter', type=int, default=C.UNFOLD_ITER, help='OmniFold iterations')
p_sy.add_argument('--ntrial', type=int, default=3, help='networks averaged per iteration')
p_sy.add_argument('--epochs', type=int, default=50, help='maximum epochs per network')
_norm(p_sy)
p_sy.add_argument('--worker', type=int, default=0, help='this worker number (0 .. n-workers-1)')
p_sy.add_argument('--n-workers', type=int, default=1, help='total workers sharing the universes')
p_sy.add_argument('--threads', type=int, default=2, help='CPUs per training')
p_sy.add_argument('--max-fail', type=int, default=3, help='stop after this many failures in a row')
p_sy.add_argument('--redo', action='store_true', help='retrain universes that are already done')
p_sy.add_argument('--dry-run', action='store_true', help='only print what would be trained')
p_sy.add_argument('--weights-base', default=C.WEIGHTS_BASE, help='output folder (default $SBND_RUNS_DIR)')
p_sy.add_argument('--universe-file', type=str, default=None,
                  help='use this weight file instead of the manifest (one source only)')
_paths(p_sy)

p_st = sub.add_parser('status', help='progress of the trainings')
p_st.add_argument('--weights-base', default=C.WEIGHTS_BASE)
p_st.add_argument('--ml-dir', default=C.ML_DIR)
p_st.add_argument('--export-dir', default=C.EXPORT_DIR)

p_ml = sub.add_parser('run-ml-unc', help='ML replicas for the network uncertainty')
p_ml.add_argument('--var', choices=C.TILT_CHOICES, default=C.EVAR, help='fake data: tilted variable')
p_ml.add_argument('--alpha', type=float, default=0.3, help='fake data: tilt strength')
p_ml.add_argument('--tag', type=str, default=None, help='fake-data tag (instead of --var/--alpha)')
p_ml.add_argument('--n-replicas', type=int, default=50)
p_ml.add_argument('--start', type=int, default=None)
p_ml.add_argument('--end', type=int, default=None)
p_ml.add_argument('--niter', type=int, default=C.UNFOLD_ITER)
p_ml.add_argument('--epochs', type=int, default=100)
p_ml.add_argument('--threads', type=int, default=2)
p_ml.add_argument('--max-fail', type=int, default=3)
p_ml.add_argument('--weights-base', default=None, help='default: $SBND_RUNS_DIR/weights_ml_unc/<tag>/')
p_ml.add_argument('--data-dir', default=C.DATA_DIR)

flags = parser.parse_args()


def _omnifold_config(data_weight_file, name, niter, ntrial, epochs, patience):
    return {
        'FILE_MC_RECO': 'mc_vals_reco.npy', 'FILE_MC_GEN': 'mc_vals_truth.npy',
        'FILE_MC_FLAG_RECO': 'mc_pass_reco.npy', 'FILE_MC_FLAG_GEN': 'mc_pass_truth.npy',
        'FILE_DATA_RECO': 'mc_vals_reco.npy', 'FILE_DATA_FLAG_RECO': 'mc_pass_reco.npy',
        'FILE_DATA_WEIGHT': data_weight_file,
        'FILE_MC_RECO_WEIGHT': 'mc_weights_reco.npy',
        'FILE_MC_GEN_WEIGHT': 'mc_weights_truth.npy',
        'NITER': niter, 'NTRIAL': ntrial, 'LR': 1e-3, 'BATCH_SIZE': 512,
        'EPOCHS': epochs, 'NAME': name, 'NPATIENCE': patience,
    }


# list-sources
def do_list_sources():
    man = C.load_manifest(flags.export_dir)
    n_ev = len(np.load(flags.data_dir + 'mc_weights_reco.npy'))
    print(f"OmniFold sample: {n_ev:,} events ({flags.data_dir})\n")
    print(f"{'key':<42s} {'family':<11s} {'label':<28s} {'rows':>6s} {'univ':>5s} {'trained':>8s}")
    bad = 0
    for fam in C.FAMILIES:
        for key, m in man.items():
            if m['family'] != fam:
                continue
            a = np.load(os.path.join(flags.export_dir, m['file']), mmap_mode='r')
            done = len(C.unfolded_push_files(key, flags.weights_base))
            ok = a.shape[0] == n_ev
            bad += not ok
            print(f"{key:<42s} {fam:<11s} {C.source_label(key):<28s} "
                  f"{a.shape[0]:>6d}{'' if ok else '!'} {a.shape[1]:>5d} {done:>8d}")
    for fam in C.FAMILIES:
        keys = [k for k, m in man.items() if m['family'] == fam]
        print(f"  {C.FAMILY_LABEL[fam]:<12s}: {len(keys):3d} sources")
    for fam, f in C.CROSSCHECK_FILES.items():
        if os.path.exists(flags.export_dir + f):
            print(f"  cross-check only (combined {C.FAMILY_LABEL[fam]} set): {f}")
    if bad:
        print(f"\n  ERROR: {bad} sources (marked !) do not have {n_ev} rows. They were exported "
              f"for a different selection; export them again after FormatData_SBND.py.")


# screen
def do_screen():
    man = C.load_manifest(flags.export_dir)
    vals, mc_w = C.load_truth(flags.data_dir)
    nom = {v: C.hist(v, vals[v], mc_w) for v in C.VARS}
    # Which sources get trained is decided on the shape part only: a pure normalisation
    # change goes through the unfolding unchanged, so the untrained estimate is
    # already right for it.
    results, shape, fam_cov = {}, {}, {}
    for key in man:
        results[key], shape[key] = {}, {}
        for kn, store in ((flags.keep_norm, results), (False, shape)):
            _, arrs = C.universe_weight_arrays(key, 'direct', mc_w, man, flags.export_dir,
                                               keep_norm=kn)
            for v in C.VARS:
                hs = np.array([C.hist(v, vals[v], mc_w * a) for a in arrs])
                cov, _ = C.covariance_from_hists(hs)
                store[key][v] = (np.sqrt(np.diag(cov)) / np.maximum(nom[v], 1e-12)).tolist()
                if store is results:
                    fk = (man[key]['family'], v)
                    fam_cov[fk] = fam_cov.get(fk, 0) + cov

    def peak(d, k):
        return max(max(d[k][v]) for v in C.VARS)

    ranked = sorted(results, key=lambda k: peak(shape, k), reverse=True)
    keep = [k for k in ranked if peak(shape, k) > flags.thresh or man[k]['family'] == 'mcstat']
    print(f"Fractional uncertainty per source from the exported weights, no training "
          f"({'with normalisation' if flags.keep_norm else 'shape only'}).\n"
          f"Sources marked * have a shape-only uncertainty above {flags.thresh:.2%} and will be trained.\n")
    print(f"{'source':<34s} {'shape':>6s} {'total':>6s}  " + '  '.join(f'{v:>30s}' for v in C.VARS))
    for k in ranked:
        cols = '  '.join(' '.join(f'{x*100:4.1f}' for x in results[k][v]).rjust(30) for v in C.VARS)
        mark = '*' if k in keep else ' '
        print(f"{mark}{C.source_label(k):<33s} {peak(shape, k)*100:5.2f}% {peak(results, k)*100:5.1f}%  {cols}")

    print(f"\nFamily totals, sources added in quadrature [%]:")
    for fam in C.FAMILIES:
        line = []
        for v in C.VARS:
            if (fam, v) in fam_cov:
                f = np.sqrt(np.diag(fam_cov[(fam, v)])) / np.maximum(nom[v], 1e-12)
                line.append(' '.join(f'{x*100:4.1f}' for x in f).rjust(30))
        print(f"  {C.FAMILY_LABEL[fam]:<12s} " + '  '.join(line))

    # compare the sum of the knobs with the combined Flux / GENIE throws
    n_ev = len(mc_w)
    for fam, fname in C.CROSSCHECK_FILES.items():
        path = flags.export_dir + fname
        if not os.path.exists(path) or np.load(path, mmap_mode='r').shape[0] != n_ev:
            continue
        man_x = {'_x': {'file': fname, 'family': fam}}
        _, arrs = C.universe_weight_arrays('_x', 'direct', mc_w, man_x, flags.export_dir,
                                           keep_norm=flags.keep_norm)
        for v in C.VARS:
            hs = np.array([C.hist(v, vals[v], mc_w * a) for a in arrs])
            comb = np.sqrt(np.diag(C.covariance_from_hists(hs)[0])).sum() / nom[v].sum()
            knob = np.sqrt(np.diag(fam_cov[(fam, v)])).sum() / nom[v].sum()
            print(f"  {C.FAMILY_LABEL[fam]:<11s} {v:<14s}: sum of knobs {knob*100:5.2f}%, "
                  f"combined throws {comb*100:5.2f}%, ratio "
                  f"{(f'{knob/comb:5.2f}' if comb > 1e-6 else 'n/a')}")

    os.makedirs(C.COV_DIR, exist_ok=True)
    with open(SCREEN_FILE, 'w') as fh:
        json.dump({'thresh': flags.thresh, 'keep_norm': flags.keep_norm, 'select_on': 'shape',
                   'keep': keep, 'frac_unc': results, 'frac_unc_shape': shape}, fh, indent=1)
    n_univ = sum(man[k]['n_univ'] for k in keep)
    print(f"\n{len(keep)} of {len(man)} sources will be trained (*): {n_univ} trainings.")
    print(f"Saved {SCREEN_FILE}")
    print(f"Train them:  python3 sbnd/RunStudies.py run-syst --source screened")
    print(f"The other sources go into the covariance untrained (BuildResults --mode hybrid).")


# check-closure
def do_check_closure():
    d = flags.closure_dir or C.closure_dir()
    PLOT_DIR = flags.plot_dir
    os.makedirs(PLOT_DIR, exist_ok=True)
    itn = C.iter_num

    push_files = sorted(glob.glob(os.path.join(d, 'Step2_Iter*_PushWeights.npy')), key=itn)
    pull_files = sorted(glob.glob(os.path.join(d, 'Step1_Iter*_PullWeights.npy')), key=itn)
    if not push_files or not pull_files:
        print(f"ERROR: No weight files in {d}. Run closure test first.")
        return

    push = C.load_push(push_files[-1])
    pull = C.load_push(pull_files[-1])
    push_bias = push.mean() - 1.0
    pull_bias = pull.mean() - 1.0

    print(f"Closure test ({d})")
    print(f"  Push: mean={push.mean():.4f}, std={push.std():.4f}, "
          f"range=[{push.min():.4f}, {push.max():.4f}]")
    print(f"  Pull: mean={pull.mean():.4f}, std={pull.std():.4f}")
    print(f"  Closure bias (push): {push_bias:+.4f} ({push_bias*100:+.2f}%)")
    print(f"  Closure bias (pull): {pull_bias:+.4f} ({pull_bias*100:+.2f}%)")

    stat_status = None
    try:
        vals, mc_weights = C.load_truth(flags.data_dir)
        if push.shape[0] != mc_weights.shape[0]:
            print(f"\n  ERROR: closure weights have {push.shape[0]} events but the current "
                  f"sample has {mc_weights.shape[0]}. Retrain: bash sbnd/runOmnifold_sbnd_closure.sh")
            raise StopIteration
        var_map = vals if flags.var == 'both' else {flags.var: vals[flags.var]}
        print(f"\n  Unfolded vs nominal truth, MC-stat errors (pass if p > {flags.pval_thresh:.2f}):")
        all_pass = True
        for vn, vv in var_map.items():
            nom_h = C.hist(vn, vv, mc_weights)
            unf_h = C.hist(vn, vv, mc_weights * push)
            var_h = C.hist(vn, vv, (mc_weights * push) ** 2)
            good = var_h > 0
            d_bin = (unf_h - nom_h)[good]
            chi2 = float(np.sum(d_bin ** 2 / var_h[good]))
            ndf = max(int(good.sum()) - 1, 1)
            pval = chi2_pvalue(chi2, ndf)
            ok = pval > flags.pval_thresh
            all_pass &= ok
            pulls = d_bin / np.sqrt(var_h[good])
            print(f"      {vn:14s}: chi2/ndf = {chi2:7.2f}/{ndf} = {chi2/ndf:6.2f}, "
                  f"p = {pval:6.3f}, max|pull| = {np.abs(pulls).max():4.2f}  "
                  f"-> {'PASS' if ok else 'FAIL'}")
        stat_status = "PASSED" if all_pass else "FAILED"
    except StopIteration:
        pass
    except FileNotFoundError:
        print(f"\n  No inputs in {flags.data_dir}, skipping the binned comparison.")

    abs_bias = abs(push_bias)
    if push.std() >= 0.2:
        print("\n  Push weights spread by more than 0.2: the networks did not converge.")
    elif abs_bias >= 0.03:
        print(f"\n  Mean push weight is {abs_bias*100:.1f}% away from 1: try more networks (--ntrial).")
    if stat_status is not None:
        print(f"\n  Closure {stat_status}")

    XLIM = (0.85, 1.15); BINS = np.linspace(*XLIM, 81)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, w, col, ttl in [(axes[0], pull, 'darkorange', 'pull, Step 1 (reco)'),
                            (axes[1], push, 'steelblue', 'push, Step 2 (truth)')]:
        ax.hist(w, bins=BINS, color=col, alpha=0.8)
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2)
        ax.set_xlim(XLIM); ax.set_xlabel('Weight'); ax.set_ylabel('Events')
        ax.set_title(f'Closure {ttl}\nmean={w.mean():.4f}, std={w.std():.4f}')
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/closure_weight_distributions.png", dpi=150); plt.close()
    print(f"  Plot saved: {PLOT_DIR}/closure_weight_distributions.png")

    if len(push_files) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        for ax, files, color, name in [(axes[0], pull_files, 'darkorange', 'pull'),
                                       (axes[1], push_files, 'steelblue', 'push')]:
            iters, means, stds = [], [], []
            for fi in files:
                w = C.load_push(fi)
                iters.append(itn(fi) + 1); means.append(w.mean()); stds.append(w.std())
            ax.errorbar(iters, means, yerr=stds, fmt='o-', color=color, capsize=3, linewidth=2)
            ax.axhline(1.0, color='red', linestyle='--')
            ax.fill_between(iters, np.array(means) - 0.01, np.array(means) + 0.01,
                            alpha=0.15, color=color, label='±1% band')
            ax.set_xlabel('OmniFold Iteration'); ax.set_ylabel(f'{name} weight mean ± std')
            ax.set_title(f"Closure {name} convergence\nbias at final iter: {means[-1]-1:+.4f}")
            ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/closure_convergence.png", dpi=150); plt.close()
        print(f"  Plot saved: {PLOT_DIR}/closure_convergence.png")


# make-fakedata
def _tilt_transform(var_name, values):
    """Standardised variable used for the tilt (log for energy/momentum)."""
    x = np.log(values.clip(1.0, None)) if var_name in ('true_p', 'true_ke') else values.astype(np.float64)
    return (x - np.mean(x)) / np.std(x)


def do_make_fakedata():
    OUT = flags.data_dir
    vals, mc_weights_reco = C.load_truth(OUT)
    n = len(mc_weights_reco)
    print(f"Loaded {n:,} events from {OUT}")
    PLOT_DIR = flags.plot_dir
    os.makedirs(PLOT_DIR, exist_ok=True)

    if flags.mode == 'tilt':
        ALPHA, VAR = flags.alpha, flags.var
        tag = make_fdt_tag(VAR, ALPHA)
        if VAR == 'both':
            tilt = ((1.0 + ALPHA * _tilt_transform(C.EVAR, vals[C.EVAR])) *
                    (1.0 + ALPHA * _tilt_transform('true_costheta', vals['true_costheta'])))
        else:
            tilt = 1.0 + ALPHA * _tilt_transform(VAR, vals[VAR])
        tilt = np.clip(tilt, 0.0, None)
        tilt = tilt * (mc_weights_reco.sum() / (mc_weights_reco * tilt).sum())
        data_weights = mc_weights_reco * tilt
        np.save(OUT + f'data_weights_sbnd_fakedata_{tag}.npy', data_weights)
        np.save(OUT + f'truth_weights_sbnd_fakedata_{tag}.npy', tilt)

        title = tilt_label(VAR, ALPHA)
        print(f"{title}, tag {tag}")
        print(f"  Tilt range: [{tilt.min():.3f}, {tilt.max():.3f}]")
        print(f"\n  Next: nohup bash sbnd/runOmnifold_sbnd_fakedata.sh --var {VAR} --alpha {ALPHA} "
              f"&   (log: logs/fdt_{tag}.log)")

        tilted = set(C.VARS) if VAR == 'both' else {VAR}
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'Fake-data injection  ({title})', fontsize=13, fontweight='bold')
        for ax, v in zip(axes, C.VARS):
            b = BINNING[v]
            ax.hist(C.fold(v, vals[v]), bins=b, weights=mc_weights_reco, alpha=0.55,
                    color='steelblue', label='Nominal MC')
            ax.hist(C.fold(v, vals[v]), bins=b, weights=data_weights, alpha=0.55,
                    color='tomato', label='Fake Data')
            ax.set_xlabel(XLABEL[v]); ax.set_ylabel('Weighted events')
            ax.set_title(f"{v}  {'← TILTED' if v in tilted else '(projected)'}")
            ax.legend()
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/fakedata_injected_{tag}.png', dpi=150); plt.close()
        print(f"  Plot: {PLOT_DIR}/fakedata_injected_{tag}.png")

    else:  # universe
        if flags.universe_file is None:
            man = C.load_manifest(flags.export_dir)
            flags.universe_file = os.path.join(flags.export_dir, man[flags.source]['file'])
        uni_all = np.load(flags.universe_file)
        if uni_all.ndim == 1:
            uni_all = uni_all[:, np.newaxis]
        assert uni_all.shape[0] == n, f"Shape mismatch: {uni_all.shape[0]} vs {n}"
        idx = flags.universe_idx
        uni_vals = np.clip(uni_all[:, idx], 0.0, 10.0)
        data_weights = C.data_weights_for_universe(mc_weights_reco, uni_all[:, idx])
        tag = f'{flags.source}_univ{idx}'
        np.save(OUT + f'data_weights_sbnd_fakedata_{tag}.npy', data_weights)
        np.save(OUT + f'truth_weights_sbnd_fakedata_{tag}.npy',
                data_weights / mc_weights_reco)
        print(f"Fake data from {C.source_label(flags.source)} universe {idx}")
        print(f"  mean={uni_vals.mean():.4f}, std={uni_vals.std():.4f}; tag = {tag}")


# run-syst
def do_run_syst():
    man = C.load_manifest(flags.export_dir)
    if flags.source == 'screened':
        if not os.path.exists(SCREEN_FILE):
            sys.exit(f"ERROR: {SCREEN_FILE} not found, run `RunStudies.py screen` first.")
        keys = json.load(open(SCREEN_FILE))['keep']
    elif flags.universe_file is not None:
        keys = [flags.source]
        man = {**man, flags.source: {'file': os.path.abspath(flags.universe_file),
                                     'family': C.family_of(flags.source, man)}}
    else:
        keys = C.resolve_sources(flags.source, man)
    if flags.only_screened:
        if not os.path.exists(SCREEN_FILE):
            sys.exit(f"ERROR: {SCREEN_FILE} not found, run `RunStudies.py screen` first.")
        kept = set(json.load(open(SCREEN_FILE))['keep'])
        skipped = [k for k in keys if k not in kept]
        keys = [k for k in keys if k in kept]
        if skipped:
            print(f"  skipping {len(skipped)} source(s) that `screen` did not choose")

    mc_weights = np.load(flags.data_dir + 'mc_weights_reco.npy')
    n_ev = len(mc_weights)

    # check every weight file before starting any training
    tasks, paths = [], {}
    for key in keys:
        path = os.path.join(flags.export_dir, man[key]['file'])
        shp = np.load(path, mmap_mode='r').shape
        if shp[0] != n_ev:
            sys.exit(f"ERROR: {path} has {shp[0]} rows, sample has {n_ev}. Re-run the notebook "
                     f"export after FormatData_SBND.py.")
        _check_settings(key)
        paths[key] = path
        end = shp[1] if flags.end is None else min(flags.end, shp[1])
        done = set(C.unfolded_push_files(key, flags.weights_base, flags.niter))
        tasks += [(key, i, flags.redo or i not in done) for i in range(flags.start, end)]
    # Mix the sources so each worker gets some of every source. The split is made
    # before dropping finished universes, so a worker always gets the same share
    # no matter when it is started.
    tasks.sort(key=lambda t: (t[1], keys.index(t[0])))
    mine = [(k, i) for k, i, todo in tasks[flags.worker::flags.n_workers] if todo]
    tasks = [t for t in tasks if t[2]]

    print(f"run-syst worker {flags.worker}/{flags.n_workers}: {len(mine)} of {len(tasks)} "
          f"remaining universes over {len(keys)} source(s)  (NITER={flags.niter}, "
          f"NTRIAL={flags.ntrial}, {'keep-norm' if flags.keep_norm else 'shape-only'}, "
          f"{flags.threads} threads)")
    if flags.dry_run:
        for key in keys:
            n = sum(1 for k, _ in mine if k == key)
            if n:
                print(f"  {key:<42s} {n:4d}")
        return

    os.makedirs(C.CONFIG_DIR, exist_ok=True)
    env = C.limit_threads(flags.threads)
    env['OMNIFOLD_WORKER'] = str(flags.worker)
    cache, n_fail, t0 = {}, 0, time.time()
    for n_done, (key, idx) in enumerate(mine):
        if key not in cache:
            cache = {key: np.load(paths[key], mmap_mode='r')}
        tag = f'{key}_univ{idx}'
        weights_dir = C.universe_dir(key, idx, flags.weights_base)
        dw_name = f'data_weights_sbnd_syst_{tag}.npy'
        config_path = f'{C.CONFIG_DIR}config_syst_{tag}.json'
        np.save(flags.data_dir + dw_name,
                C.data_weights_for_universe(mc_weights, np.asarray(cache[key][:, idx]),
                                            flags.keep_norm))
        C.write_omnifold_config(config_path, _omnifold_config(
            dw_name, f'sbnd_syst_{tag}', flags.niter, flags.ntrial, flags.epochs, 7))
        os.makedirs(weights_dir, exist_ok=True)
        el = (time.time() - t0) / 60
        eta = el / n_done * (len(mine) - n_done) / 60 if n_done else float('nan')
        print(f"\n[{n_done+1}/{len(mine)}] {C.source_label(key)} universe {idx}   "
              f"(elapsed {el:.0f} min, ETA {eta:.1f} h)", flush=True)
        rc = subprocess.run([sys.executable, 'run_sbnd.py', '--config', config_path,
                             '--file_path', flags.data_dir, '--weights_folder', weights_dir,
                             '--plot_folder', weights_dir, '--no_eff', '--verbose'],
                            env=env).returncode
        for f in (config_path, flags.data_dir + dw_name):
            if os.path.exists(f):
                os.remove(f)
        _drop_models(weights_dir)
        ok = rc == 0 and idx in C.unfolded_push_files(key, flags.weights_base, flags.niter)
        if ok:
            n_fail = 0
            continue
        n_fail += 1
        with open(C.FAILED_LOG, 'a') as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {tag}  rc={rc}\n")
        print(f"  FAILED: {tag} (exit code {rc}), logged in {C.FAILED_LOG}", flush=True)
        if n_fail >= flags.max_fail:
            sys.exit(f"ERROR: {n_fail} consecutive failures, stopping this worker. Usually the "
                     f"per-user thread/process limit: lower --n-workers or --threads, then "
                     f"relaunch (finished universes are skipped).")
    print("\nAll universes of this worker complete.")


def _drop_models(folder):
    """Remove the Keras checkpoints (*.h5); only the push/pull .npy files are used later."""
    for f in glob.glob(os.path.join(folder, '*.h5')):
        try:
            os.remove(f)
        except OSError:
            pass


def _check_settings(key):
    """All universes of a source must use the same normalisation and iteration settings."""
    os.makedirs(f'{flags.weights_base}/weights_{key}', exist_ok=True)
    sp = C.run_settings_path(key, flags.weights_base)
    want = {'keep_norm': bool(flags.keep_norm), 'niter': flags.niter,
            'ntrial': flags.ntrial, 'epochs': flags.epochs}
    prev = None
    if os.path.exists(sp):
        try:
            with open(sp) as fh:
                prev = json.load(fh)
        except (json.JSONDecodeError, OSError):
            prev = None
    if prev is not None and not flags.redo:
        if (bool(prev.get('keep_norm')) != want['keep_norm']
                or prev.get('niter') != want['niter']):
            sys.exit(f"ERROR: {key} already has universes trained with keep_norm="
                     f"{prev.get('keep_norm')}, niter={prev.get('niter')}. Use the same "
                     f"settings, or delete {flags.weights_base}/weights_{key}/ to retrain.")
        return
    tmp = f'{sp}.{os.getpid()}.tmp'
    with open(tmp, 'w') as fh:
        json.dump(want, fh)
    os.replace(tmp, sp)


# status
def do_status():
    man = C.load_manifest(flags.export_dir)
    keys = json.load(open(SCREEN_FILE))['keep'] if os.path.exists(SCREEN_FILE) else list(man)
    done_tot = want_tot = 0
    recent, now = 0, time.time()
    print(f"{'source':<34s} {'done':>9s}")
    for key in keys:
        files = C.unfolded_push_files(key, flags.weights_base)
        n = man[key].get('n_univ', 100)
        done_tot += len(files); want_tot += n
        recent += sum(1 for f in files.values() if now - os.path.getmtime(f) < 3600)
        bar = '#' * int(20 * len(files) / n)
        print(f"{C.source_label(key):<34s} {len(files):4d}/{n:<4d} {bar}")
    ml = {}
    for d in glob.glob(flags.ml_dir + '*/replica_*/'):
        if glob.glob(d + f'Step2_Iter{C.UNFOLD_ITER-1}_*_PushWeights.npy'):
            tag = os.path.basename(os.path.dirname(os.path.dirname(d)))
            ml[tag] = ml.get(tag, 0) + 1
    ml_txt = ', '.join(f'{t}: {n}' for t, n in sorted(ml.items())) or 'none'
    print(f"\nSyst universes: {done_tot}/{want_tot} ({100*done_tot/max(want_tot,1):.1f}%)   "
          f"ML replicas done: {ml_txt}")
    if recent:
        print(f"Rate (last hour): {recent} universes/h  ->  ETA {(want_tot-done_tot)/recent:.1f} h")
    if os.path.exists(C.FAILED_LOG):
        lines = open(C.FAILED_LOG).read().splitlines()
        print(f"Failed trainings: {len(lines)} (last: {lines[-1] if lines else '-'}). "
              f"Relaunching retries them.")
    try:
        ps = subprocess.run(['pgrep', '-fc', 'run_sbnd.py'], capture_output=True, text=True)
        print(f"OmniFold trainings running now: {ps.stdout.strip() or 0}")
    except FileNotFoundError:
        pass


# run-ml-unc
def do_run_ml_unc():
    fdt_tag = flags.tag or make_fdt_tag(flags.var, flags.alpha)
    flags.weights_base = flags.weights_base or C.ml_dir(fdt_tag)
    replica_range = (range(flags.start, flags.end)
                     if flags.start is not None and flags.end is not None
                     else range(flags.n_replicas))
    os.makedirs(flags.weights_base, exist_ok=True)
    os.makedirs(C.CONFIG_DIR, exist_ok=True)
    n_fail = 0
    for idx in replica_range:
        tag = f'replica_{idx}'
        weights_dir = f'{flags.weights_base}/{tag}/'
        final = glob.glob(weights_dir + f'Step2_Iter{flags.niter-1}_*_PushWeights.npy')
        if final:
            continue
        os.makedirs(weights_dir, exist_ok=True)
        config_path = f'{C.CONFIG_DIR}config_ml_unc_{tag}.json'
        C.write_omnifold_config(config_path, _omnifold_config(
            f'data_weights_sbnd_fakedata_{fdt_tag}.npy', f'sbnd_ml_unc_{tag}',
            flags.niter, 1, flags.epochs, 10))
        print(f"\nML replica {idx} (NTRIAL=1, NITER={flags.niter})", flush=True)
        env = C.limit_threads(flags.threads)
        env['TF_RANDOM_SEED'] = str(42 + idx * 137)
        env['PYTHONHASHSEED'] = str(idx)
        env['OMNIFOLD_WORKER'] = str(1000 + idx)
        rc = subprocess.run([sys.executable, 'run_sbnd.py', '--config', config_path,
                             '--file_path', flags.data_dir, '--weights_folder', weights_dir,
                             '--plot_folder', weights_dir, '--no_eff', '--verbose'],
                            env=env).returncode
        if os.path.exists(config_path):
            os.remove(config_path)
        _drop_models(weights_dir)
        if rc == 0 and glob.glob(weights_dir + f'Step2_Iter{flags.niter-1}_*_PushWeights.npy'):
            n_fail = 0
            continue
        n_fail += 1
        print(f"  FAILED: {tag} (exit code {rc})", flush=True)
        if n_fail >= flags.max_fail:
            sys.exit(f"ERROR: {n_fail} consecutive failures, stopping. Relaunch with fewer "
                     f"parallel jobs; finished replicas are skipped.")
    print(f"\nML replicas done. Next: python3 sbnd/BuildResults.py covariance --source ml --ml-tag {fdt_tag}")


{'list-sources':  do_list_sources,
 'status':        do_status,
 'screen':        do_screen,
 'check-closure': do_check_closure,
 'make-fakedata': do_make_fakedata,
 'run-syst':      do_run_syst,
 'run-ml-unc':    do_run_ml_unc}[flags.action]()
