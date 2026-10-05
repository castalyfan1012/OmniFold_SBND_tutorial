"""
Plots for the fake-data validation and for the final results.

    python3 sbnd/MakePlots.py validation [options]
    python3 sbnd/MakePlots.py results [options]
    python3 sbnd/MakePlots.py all [options]

validation  (sbnd/plots_validation/)
    Weight recovery, unfolded vs injected distributions with the chi2/ndf,
    chi2/ndf vs iteration, and a pass/fail verdict (p-value of the stat-only
    chi2 above --pval-thresh).

results  (sbnd/plots_xsec/ and sbnd/plots_syst/)
    Cross section vs the injected truth (xsec_ml_*, needs at least 2 ML
    replicas for this tag), uncertainty budget by family with and without the
    normalisation, the leading sources inside each family, correlation
    matrices, 2D slices and the 2D correlation. Run BuildResults.py
    covariance and xsec first.

all
    Both of the above.

Both kinematic variables are always plotted; --var and --alpha only pick
which fake-data sample (tag) to use.

Examples:
    python3 sbnd/MakePlots.py validation --var true_p --alpha 0.3
    python3 sbnd/MakePlots.py results --var true_p --alpha 0.3 --cov-source all
    python3 sbnd/MakePlots.py all --var both --alpha 0.3
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sbnd_config as C
from sbnd_config import (BINNING, XLABEL, YLABEL_XSEC, FAMILY_LABEL, FAMILY_COLOR,
                         chi2_pvalue, fit_annotation, make_fdt_tag, tilt_label)

mpl.rcParams.update({
    'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
    'figure.dpi': 150, 'axes.grid': False,
})

parser = argparse.ArgumentParser(description='SBND OmniFold plots.')
sub = parser.add_subparsers(dest='action', required=True)


def _common(p):
    p.add_argument('--var', choices=C.TILT_CHOICES, default=C.EVAR,
                   help='variable that was tilted in the fake data (sets the tag)')
    p.add_argument('--alpha', type=float, default=0.3, help='tilt strength of the fake data')
    p.add_argument('--tag', default=None, help='give the fake-data tag directly instead of --var/--alpha')
    p.add_argument('--data-dir', default=C.DATA_DIR, help='formatted data (default $SBND_DATA_DIR)')
    p.add_argument('--cov-dir', default=C.COV_DIR, help='where BuildResults.py wrote the covariances')
    p.add_argument('--iter', type=int, default=C.UNFOLD_ITER, help='OmniFold iteration to plot')
    p.add_argument('--iter-rel-tol', type=float, default=0.05,
                   help='relative chi2 change used to say the iterations have converged')


def _validation(p):
    p.add_argument('--weights-dir', default=None,
                   help='fake-data training (default sbnd/runs/weights_sbnd_fakedata_<tag>/)')
    p.add_argument('--val-dir', default='sbnd/plots_validation',
                   help='output folder for the validation plots')
    p.add_argument('--fds-cov-source', default='fds',
                   help='covariance group drawn as the band in the validation plots')
    p.add_argument('--pval-thresh', type=float, default=0.05, help='p-value needed to pass')
    p.add_argument('--extended', action='store_true',
                   help='also plot the chi2 convergence without the lowest bin')


def _results(p):
    p.add_argument('--weights-base', default=C.WEIGHTS_BASE, help='systematic universe trainings')
    p.add_argument('--export-dir', default=C.EXPORT_DIR, help='universe weights (default $SBND_EXPORT_DIR)')
    p.add_argument('--plot-dir', default='sbnd/plots_xsec', help='output folder for the cross-section plots')
    p.add_argument('--syst-dir', default='sbnd/plots_syst', help='output folder for the uncertainty plots')
    p.add_argument('--cov-source', default='all', help='covariance group for the bands (all, fds, xsec, syst)')
    p.add_argument('--mode', choices=['hybrid', 'unfolded', 'direct'], default='hybrid',
                   help='universes used for the 2D correlation, same as in BuildResults.py')
    p.add_argument('--top-n', type=int, default=8, help='number of sources shown per family breakdown')
    p.add_argument('--ml-weights-dir', default=None,
                   help='ML replicas (default sbnd/runs/weights_ml_unc/<tag>/)')
    p.add_argument('--wsvd-syst-file', default=None,
                   help='optional JSON {var: {family: [frac unc per bin]}} from Wiener-SVD to overlay')


p_val = sub.add_parser('validation', help='fake-data validation plots')
_common(p_val); _validation(p_val)
p_res = sub.add_parser('results', help='cross section and uncertainty plots')
_common(p_res); _results(p_res)
p_res.add_argument('--val-dir', default='sbnd/plots_validation', help='where the chi2 convergence plot goes')
p_all = sub.add_parser('all', help='validation and results')
_common(p_all); _validation(p_all); _results(p_all)

flags = parser.parse_args()
TAG = flags.tag or make_fdt_tag(flags.var, flags.alpha)
if getattr(flags, 'ml_weights_dir', 'x') is None:
    flags.ml_weights_dir = C.ml_dir(TAG)
TILT_DESC = tilt_label(flags.var, flags.alpha)
BUDGET_FAMILIES = ['flux', 'genie', 'extra_xsec', 'g4', 'mcstat', 'ml']


CHI2_YLIM = (1e-2, 1e3)   # same range on every convergence plot so they can be compared
# the momentum spectrum falls, so its top right is empty; cos(theta) rises towards 1
LEGEND_SIDE = {C.EVAR: 'right', 'true_costheta': 'left'}


def xlabel(vn):
    return XLABEL[vn]


def set_bin_ticks(ax, vn, axis='x'):
    """Bin-edge ticks, '3000+' on the overflow edge, compressed wide cosθ bin."""
    C.apply_axis(ax, vn, axis=axis, label=False)


def legend_with_box(ax, text, side, fontsize=9, headroom=1.55, **legend_kw):
    """Legend in the top corner on `side`, fit-summary box directly below it.
    Extends the y-range so neither covers the histogram."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, lo + (hi - lo) * headroom)
    loc = f'upper {side}'
    leg = ax.legend(loc=loc, fontsize=fontsize, framealpha=0.9, **legend_kw)
    ax.figure.canvas.draw()
    bb = leg.get_window_extent().transformed(ax.transAxes.inverted())
    x = bb.x1 if side == 'right' else bb.x0
    ax.text(x, bb.y0 - 0.02, text, transform=ax.transAxes, fontsize=fontsize,
            va='top', ha=side, multialignment=side)
    return leg


def plot_chi2_curve(ax, xs, ys, **kw):
    """chi2/ndf vs iteration on the fixed CHI2_YLIM range; points outside the
    range are pinned to the edge and drawn as open triangles."""
    ys = np.asarray(ys, float); xs = np.asarray(xs)
    lo, hi = CHI2_YLIM
    yc = np.clip(ys, lo * 1.05, hi / 1.05)
    ax.plot(xs, yc, 'o-', **kw)
    col = kw.get('color')
    for m, mk in ((ys < lo, 'v'), (ys > hi, '^')):
        if m.any():
            ax.plot(xs[m], yc[m], mk, color=col, markerfacecolor='white', markersize=9)


def finish_chi2_axes(ax):
    ax.set_yscale('log'); ax.set_ylim(*CHI2_YLIM)
    ax.axhline(1.0, color='gray', linestyle=':', label=r'$\chi^2$/DoF = 1')
    ax.set_xlabel('OmniFold Iteration'); ax.set_ylabel(r'$\chi^2$/DoF')


def dedup_push_files(files):
    by_iter = {}
    for f in files:
        by_iter.setdefault(C.iter_num(f), []).append(f)
    return [sorted(v)[-1] for k, v in sorted(by_iter.items())]


def chi2_simple(obs, exp):
    m = exp > 0
    return float(np.sum((obs[m] - exp[m]) ** 2 / exp[m]))


def cov_chi2(unf, truth, cov):
    d = np.asarray(unf) - np.asarray(truth)
    try:
        return float(d @ np.linalg.inv(cov) @ d)
    except np.linalg.LinAlgError:
        return float(np.sum(d ** 2 / np.diag(cov).clip(1e-30)))


def recommend_iteration(iters, chi2ndf, rel_tol=0.05):
    it = np.asarray(iters); cv = np.asarray(chi2ndf, float)
    m = it > 0; itv, cvv = it[m], cv[m]
    if len(cvv) == 0:
        return None, None
    gmin = int(itv[int(np.nanargmin(cvv))]); plateau = int(itv[-1])
    for k in range(1, len(cvv)):
        if np.isfinite(cvv[k]) and np.isfinite(cvv[k-1]):
            if (cvv[k-1] - cvv[k]) / max(abs(cvv[k-1]), 1e-12) < rel_tol:
                plateau = int(itv[k]); break
    return gmin, plateau


def load_cov(name, vn):
    path = f'{flags.cov_dir}/covariance_{name}_{vn}.npz'
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=False)
    if d['cov'].shape[0] != len(BINNING[vn]) - 1:
        print(f"    WARNING: {path} has the old binning, rebuild with BuildResults covariance")
        return None
    return d


def frac_unc_of(d, family=None):
    cov = d['cov'].copy()
    if family == 'ml' and 'n_universes' in d and int(d['n_universes']) > 1:
        cov = cov / int(d['n_universes'])          # central value = replica mean
    return np.sqrt(np.diag(cov)) / np.maximum(d['mean_hist'], 1e-12)


def step(ax, vn, y, **kw):
    b = BINNING[vn]
    ax.step(b, np.append(y, y[-1]), where='post', **kw)


def load_fakedata(weights_dir=None):
    vals, mc_w = C.load_truth(flags.data_dir)
    tilt_path = flags.data_dir + f'truth_weights_sbnd_fakedata_{TAG}.npy'
    if not os.path.exists(tilt_path):
        print(f"ERROR: {tilt_path} not found. Run: python3 sbnd/RunStudies.py make-fakedata "
              f"--mode tilt --var {flags.var} --alpha {flags.alpha}")
        return None
    wd = weights_dir or C.fakedata_dir(TAG)
    push_files = dedup_push_files(sorted(glob.glob(wd + 'Step2_Iter*_PushWeights.npy'),
                                         key=C.iter_num))
    if not push_files:
        print(f"ERROR: no push files in '{wd}', train with runOmnifold_sbnd_fakedata.sh")
        return None
    if len(C.load_push(push_files[-1])) != len(mc_w):
        print(f"ERROR: push weights in '{wd}' have a different event count than the current "
              f"sample ({len(mc_w)}). Retrain the fake-data run after FormatData_SBND.py.")
        return None
    return vals, mc_w, np.load(tilt_path), push_files


# validation
def do_validation(plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    loaded = load_fakedata(getattr(flags, 'weights_dir', None))
    if loaded is None:
        return
    vals, mc_w, injected, push_files = loaded
    push_mean = C.load_push(C.pick_iter(push_files, flags.iter))
    print(f"Validation of {TAG} (trained {C.iter_num(push_files[-1]) + 1} iterations; "
          f"results at iteration {C.iter_num(C.pick_iter(push_files, flags.iter)) + 1})")
    print(f"  Push: mean={push_mean.mean():.4f}, std={push_mean.std():.4f}")

    # weight recovery
    for vn in C.VARS:
        b = BINNING[vn]; cen = 0.5 * (b[:-1] + b[1:])
        x = C.fold(vn, vals[vn])
        def bmean(w):
            out = np.zeros(len(b) - 1)
            for i in range(len(b) - 1):
                m = (x >= b[i]) & (x < b[i+1])
                out[i] = w[m].mean() if m.any() else np.nan
            return out
        pb, tb = bmean(push_mean), bmean(injected)
        fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1]})
        cen = C.apply_index_axis(axes[1], vn)          # equal-width bins
        axes[0].plot(cen, tb, 'b-o', label='Injected')
        axes[0].plot(cen, pb, 'r-s', label='OmniFold push')
        axes[0].axhline(1.0, color='gray', linestyle='--')
        axes[0].set_ylabel('Mean weight'); axes[0].legend()
        axes[0].set_title(f'Recovery: {vn}  ({TILT_DESC})')
        axes[1].plot(cen, pb / np.where(tb > 0, tb, 1.0), 'k-o')
        axes[1].axhline(1.0, color='gray', linestyle='--')
        axes[1].fill_between([0, len(cen)], 0.8, 1.2, alpha=0.15, color='green')
        axes[1].set_ylabel('Push / Injected'); axes[1].set_ylim(0.5, 1.5)
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/fakedata_{TAG}_recovery_{vn}.png'); plt.close()
        print(f"  fakedata_{TAG}_recovery_{vn}.png")

    # unfolded distributions + verdict
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    status = {}
    for ax, vn in zip(axes, C.VARS):
        b = BINNING[vn]; cen = 0.5 * (b[:-1] + b[1:]); n_bins = len(b) - 1
        nom_h = C.hist(vn, vals[vn], mc_w)
        fd_h = C.hist(vn, vals[vn], mc_w * injected)
        unf_h = C.hist(vn, vals[vn], mc_w * push_mean)
        unf_w2 = C.hist(vn, vals[vn], (mc_w * push_mean) ** 2)
        stat_err = np.sqrt(np.maximum(unf_w2, 0))
        cd = load_cov(flags.fds_cov_source, vn)
        c2 = cov_chi2(unf_h, fd_h, np.diag(unf_w2.clip(1e-30)))
        ndf = n_bins - 1
        status[vn] = (c2, ndf, chi2_pvalue(c2, ndf))
        step(ax, vn, nom_h, color='gray', linewidth=1.5, linestyle='--', label='Nominal MC')
        step(ax, vn, fd_h, color='black', linewidth=2, label='Tilted data')
        if cd is not None:
            band = np.sqrt(np.diag(cd['cov']))
            for i in range(n_bins):
                ax.fill_between([b[i], b[i+1]], unf_h[i] - band[i], unf_h[i] + band[i],
                                color='red', alpha=0.15,
                                label=f'{flags.fds_cov_source} band' if i == 0 else None)
        ax.errorbar(cen, unf_h, yerr=stat_err, fmt='ro', markersize=5, capsize=3, label='OmniFold')
        ax.set_xlabel(xlabel(vn)); ax.set_ylabel('Weighted events'); set_bin_ticks(ax, vn)
        ax.set_title(f'Unfolded: {vn}  ({TILT_DESC})', fontsize=12)
        ax.set_ylim(bottom=0)
        legend_with_box(ax, fit_annotation(c2, ndf, '[stat cov]'), LEGEND_SIDE[vn])
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/fakedata_{TAG}_unfolded_distributions.png'); plt.close()

    thr = getattr(flags, 'pval_thresh', 0.05)
    print(f"\n  Fake-data test (stat cov, pass if p > {thr:.2f})")
    ok_all = True
    for vn, (c2, ndf, p) in status.items():
        ok = p > thr; ok_all &= ok
        print(f"    {vn:14s}: chi2/ndf = {c2:6.2f}/{ndf} = {c2/ndf:5.2f}, p = {p:6.3f} "
              f"-> {'PASS' if ok else 'FAIL'}")
    print(f"  Fake-data test {'passed' if ok_all else 'FAILED'}")

    # chi2/ndf vs iteration
    _chi2_convergence(vals, mc_w, injected, push_files, f'{plot_dir}/fakedata_{TAG}_chi2_convergence.png')

    if getattr(flags, 'extended', False):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, n_excl in zip(axes, (1, 2)):
            for vn, col in zip(C.VARS, ('red', 'blue')):
                keep = list(range(n_excl, len(BINNING[vn]) - 1))
                if len(keep) < 2:
                    continue
                th = C.hist(vn, vals[vn], mc_w * injected)
                ys, xs = [], [0]
                ys.append(chi2_simple(C.hist(vn, vals[vn], mc_w)[keep], th[keep]) / (len(keep) - 1))
                for f in push_files:
                    h = C.hist(vn, vals[vn], mc_w * C.load_push(f))
                    xs.append(C.iter_num(f) + 1); ys.append(chi2_simple(h[keep], th[keep]) / (len(keep) - 1))
                plot_chi2_curve(ax, xs, ys, color=col, label=C.SHORT[vn])
            finish_chi2_axes(ax)
            ax.set_title(f'excluding lowest {n_excl} bin(s)'); ax.legend()
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/fakedata_{TAG}_chi2_exclude_bins.png'); plt.close()

    # per-bin chi2 contribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, vn in zip(axes, C.VARS):
        b = BINNING[vn]; n_bins = len(b) - 1
        th = C.hist(vn, vals[vn], mc_w * injected); uh = C.hist(vn, vals[vn], mc_w * push_mean)
        per_bin = np.where(th > 0, (uh - th) ** 2 / np.maximum(th, 1e-12), 0)
        ax.bar(np.arange(n_bins), per_bin, color='steelblue', alpha=0.7, edgecolor='navy')
        ax.set_xticks(np.arange(n_bins)); ax.set_xticklabels(C.bin_labels(vn), rotation=30, fontsize=9)
        ax.set_ylabel(r'Per-bin $\chi^2$'); ax.set_title(fit_annotation(per_bin.sum(), n_bins - 1).replace('\n', ',  '), fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/fakedata_{TAG}_chi2_bin_diagnostic.png'); plt.close()
    print(f"  plots -> {plot_dir}")


def _chi2_convergence(vals, mc_w, injected, push_files, out):
    """chi2/ndf of the unfolded spectrum vs the injected truth at every iteration.
    Same statistic as the fake-data verdict: diagonal stat cov = Σ(w·push)² per bin,
    ndf = n_bins - 1 (total preserved). Iteration 0 = prior (push = 1)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    table = {}
    for vn, col in zip(C.VARS, ('red', 'blue')):
        ndf = len(BINNING[vn]) - 2
        th = C.hist(vn, vals[vn], mc_w * injected)
        iters, c2s = [], []
        for it, push in [(0, np.ones_like(mc_w))] + \
                        [(C.iter_num(f) + 1, C.load_push(f)) for f in push_files]:
            uh = C.hist(vn, vals[vn], mc_w * push)
            var = C.hist(vn, vals[vn], (mc_w * push) ** 2)
            g = var > 0
            iters.append(it); c2s.append(float(np.sum((uh[g] - th[g]) ** 2 / var[g])))
        table[vn] = (iters, c2s, ndf)
        plot_chi2_curve(ax, iters, np.array(c2s) / ndf, color=col, linewidth=2, label=C.SHORT[vn])
    finish_chi2_axes(ax)
    ax.set_title(rf'$\chi^2$ convergence ({TILT_DESC})'); ax.legend()
    plt.tight_layout(); plt.savefig(out); plt.close()

    # per-iteration printout
    print(f"\n  chi2/ndf vs iteration (stat cov; iteration 0 = prior; p in brackets)")
    print(f"  {'iter':>5s}" + ''.join(f"{vn:>26s}" for vn in table))
    for k, it in enumerate(table[C.VARS[0]][0]):
        row = ''
        for vn, (iters, c2s, ndf) in table.items():
            c2 = c2s[k]
            row += f"{c2:9.2f}/{ndf} = {c2/ndf:7.3f} ({chi2_pvalue(c2, ndf):5.3f})"
        tag = '  <- used' if it == getattr(flags, 'iter', C.UNFOLD_ITER) else ''
        print(f"  {it:5d}  {row}{tag}")


# results
def do_results():
    for d in (flags.plot_dir, flags.syst_dir, flags.val_dir):
        os.makedirs(d, exist_ok=True)
    loaded = load_fakedata()
    if loaded is None:
        return
    vals, mc_w, injected, push_files = loaded

    # central value: mean of the ML replicas when there are any
    reps = []
    for rdir in sorted(glob.glob(flags.ml_weights_dir + 'replica_*/')):
        pf = C.pick_iter(glob.glob(rdir + 'Step2_Iter*_PushWeights.npy'), flags.iter)
        if pf:
            w = C.load_push(pf)
            if len(w) == len(mc_w):
                reps.append(w)
    push_final = (np.mean(reps, axis=0) if len(reps) >= 2
                  else C.load_push(C.pick_iter(push_files, flags.iter)))
    print(f"  Central result: {'mean of %d ML replicas' % len(reps) if len(reps) >= 2 else 'single fake-data training'}"
          f" ({flags.ml_weights_dir if len(reps) >= 2 else C.fakedata_dir(TAG)})")

    man = C.load_manifest(flags.export_dir)
    for vn in C.VARS:
        if len(reps) >= 2:
            _xsec_vs_truth(vn, vals, mc_w, injected, push_final, len(reps))
        elif vn == C.VARS[0]:
            print(f"  No ML replicas for {TAG}: xsec_ml_* skipped "
                  f"(the single-training version is made by BuildResults.py xsec)")
        _budget(vn)
        _family_breakdowns(vn, man)
        _correlations(vn)
    _chi2_convergence(vals, mc_w, injected, push_files, f'{flags.val_dir}/chi2_convergence_{TAG}.png')
    _snapshots(vals, mc_w, injected, push_files)
    _weights(vals, mc_w, injected, push_final, push_files)
    cov2d = _cov_2d(vals, mc_w, man, BINNING[C.EVAR], BINNING['true_costheta'])
    _slices_2d(vals, mc_w, injected, push_final, cov2d)
    _correlation_2d(vals, mc_w, man)
    print(f"\nResults plots -> {flags.plot_dir}, budgets -> {flags.syst_dir}")


def _band(ax, vn, y, err, **kw):
    b = BINNING[vn]
    for i in range(len(b) - 1):
        ax.fill_between([b[i], b[i + 1]], y[i] - err[i], y[i] + err[i],
                        label=kw.pop('label', None) if i == 0 else None, **kw)


def _xsec_vs_truth(vn, vals, mc_w, injected, push, n_rep=0):
    b = BINNING[vn]; n_bins = len(b) - 1; cen = 0.5 * (b[:-1] + b[1:])
    eff = C.load_efficiency(vn, flags.export_dir, flags.data_dir)
    scale = eff.clip(1e-6) * np.diff(b)
    truth = C.hist(vn, vals[vn], mc_w * injected) / scale
    nom = C.hist(vn, vals[vn], mc_w) / scale
    unf = C.hist(vn, vals[vn], mc_w * push) / scale
    stat = np.sqrt(C.hist(vn, vals[vn], (mc_w * push) ** 2)) / scale
    cd = load_cov(flags.cov_source, vn)
    tot = np.sqrt(np.diag(cd['cov'])) / scale if cd is not None else None
    c2 = float(np.sum((unf - truth) ** 2 / (stat ** 2).clip(1e-30)))
    gof = fit_annotation(c2, n_bins - 1, '[stat cov]')
    print(f"\n  {vn}: xsec vs truth (stat cov): {gof.replace(chr(10), ', ')}")

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    if tot is not None:
        _band(axes[0], vn, unf, tot, color='red', alpha=0.15, linewidth=0,
              label=f'Total unc. ({flags.cov_source})')
    step(axes[0], vn, truth, color='black', linewidth=2, label='Tilted data')
    step(axes[0], vn, nom, color='gray', linewidth=1.5, linestyle='--', label='Prior')
    who = f'mean of {n_rep} ML replicas' if n_rep else 'single training'
    axes[0].errorbar(cen, unf, yerr=stat, fmt='o', color='red', capsize=3, label=f'OmniFold, {who} (stat.)')
    axes[0].set_ylabel(YLABEL_XSEC[vn])
    axes[0].set_title(rf'SBND $\nu_e$ CC Inclusive  ({TILT_DESC}, {who})', fontsize=11)
    axes[0].set_ylim(bottom=0)
    legend_with_box(axes[0], gof, LEGEND_SIDE[vn])
    r = np.where(truth > 0, truth, 1)
    if tot is not None:
        _band(axes[1], vn, unf / r, tot / r, color='red', alpha=0.15, linewidth=0)
    axes[1].errorbar(cen, unf / r, yerr=stat / r, fmt='o', color='red', capsize=3)
    axes[1].axhline(1.0, color='black')
    axes[1].set_ylim(0.5, 1.5); axes[1].set_ylabel('OmniFold / Tilted')
    axes[1].set_xlabel(xlabel(vn)); set_bin_ticks(axes[1], vn)
    plt.tight_layout()
    kind = 'ml' if n_rep else 'fdt'
    plt.savefig(f'{flags.plot_dir}/xsec_{kind}_{TAG}_{vn}.png'); plt.close()
    print(f"    xsec_{kind}_{TAG}_{vn}.png")


def _budget(vn):
    """Family-level fractional uncertainty and total, with and without normalisation."""
    fams = {}
    for fam in BUDGET_FAMILIES:
        d = load_cov(fam, vn)
        if d is not None:
            cov = d['cov'] / int(d['n_universes']) if (fam == 'ml' and 'n_universes' in d
                                                        and int(d['n_universes']) > 1) else d['cov']
            fams[fam] = (cov, d)
    tot = load_cov(flags.cov_source, vn)
    wsvd = {}
    if flags.wsvd_syst_file and os.path.exists(flags.wsvd_syst_file):
        raw = json.load(open(flags.wsvd_syst_file))
        raw = raw.get(vn, {})
        wsvd = {C.LEGACY_FAMILY.get(k, k): np.array(v) for k, v in raw.items()}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, shape, ttl in ((axes[0], False, 'with normalization'),
                           (axes[1], True, 'without normalization (shape only)')):
        for fam, (cov, d) in fams.items():
            ref = d['mean_hist']
            c = shape_cov(cov, ref) if shape else cov
            note = ''
            if 'n_sources' in d:
                note = f" ({int(d['n_sources'])} src)"
            elif fam == 'ml' and 'n_universes' in d:
                note = f" (σ/√{int(d['n_universes'])})"
            step(ax, vn, np.sqrt(np.diag(c)) / np.maximum(ref, 1e-12), color=FAMILY_COLOR[fam],
                 linewidth=1.8, label=FAMILY_LABEL[fam] + note)
        if tot is not None:
            c = shape_cov(tot['cov'], tot['mean_hist']) if shape else tot['cov']
            step(ax, vn, np.sqrt(np.diag(c)) / np.maximum(tot['mean_hist'], 1e-12), color='black',
                 linewidth=2.5, label=f'Total ({flags.cov_source})')
        if not shape:
            for fam, y in wsvd.items():
                if fam in FAMILY_COLOR and len(y) == len(BINNING[vn]) - 1:
                    step(ax, vn, y, color=FAMILY_COLOR[fam], linestyle='--', linewidth=1.2,
                         label=f'W-SVD {FAMILY_LABEL[fam]}')
        ax.set_yscale('log'); ax.set_ylim(1e-3, 1.0)
        ax.set_xlabel(xlabel(vn)); set_bin_ticks(ax, vn)
        ax.set_title(ttl, fontsize=13)
        ax.legend(fontsize=8, ncol=2 if (wsvd and not shape) else 1, loc='upper right')
    axes[0].set_ylabel('Fractional uncertainty')
    plt.suptitle(rf'Uncertainty budget: SBND $\nu_e$ CC, {vn}')
    plt.tight_layout()
    plt.savefig(f'{flags.syst_dir}/uncertainty_budget_{vn}.png'); plt.close()
    out = {f: frac_unc_of(d, f) for f, (_, d) in fams.items()}
    with open(f'{flags.cov_dir}/omnifold_source_frac_unc_{vn}.json', 'w') as fh:
        json.dump({vn: {f: y.tolist() for f, y in out.items()},
                   'labels': {f: FAMILY_LABEL[f] for f in out}}, fh, indent=2)
    print(f"    uncertainty_budget_{vn}.png")


def _family_breakdowns(vn, man):
    """Leading individual sources inside each family (tag-independent)."""
    for fam in ['flux', 'genie', 'extra_xsec', 'g4']:
        keys = [k for k, m in man.items() if m['family'] == fam]
        curves = []
        for k in keys:
            d = load_cov(k, vn)
            if d is not None:
                curves.append((frac_unc_of(d).max(), k, frac_unc_of(d), str(d['mode']) if 'mode' in d else ''))
        if not curves:
            continue
        curves.sort(reverse=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        fd = load_cov(fam, vn)
        if fd is not None:
            step(ax, vn, frac_unc_of(fd), color='black', linewidth=2.5,
                 label=f'{FAMILY_LABEL[fam]} total ({len(curves)} src)')
        cmap = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, (_, k, f, mode) in enumerate(curves[:flags.top_n]):
            step(ax, vn, f, color=cmap[i % 10], linewidth=1.5,
                 linestyle='-' if mode != 'direct' else '--',
                 label=C.source_label(k) + (' [direct]' if mode == 'direct' else ''))
        rest = curves[flags.top_n:]
        if rest:
            rc = sum(np.diag(load_cov(k, vn)['cov']) for _, k, _, _ in rest)
            ref = load_cov(fam, vn)['mean_hist'] if fd is not None else None
            if ref is not None:
                step(ax, vn, np.sqrt(rc) / np.maximum(ref, 1e-12), color='gray', linewidth=1.2,
                     linestyle=':', label=f'other {len(rest)} (quad. sum)')
        ax.set_yscale('log'); ax.set_ylim(1e-4, 1.0)
        ax.set_xlabel(xlabel(vn)); ax.set_ylabel('Fractional uncertainty'); set_bin_ticks(ax, vn)
        ax.set_title(f'{FAMILY_LABEL[fam]}: leading sources ({vn})')
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(f'{flags.syst_dir}/uncertainty_breakdown_{fam}_{vn}.png'); plt.close()
        print(f"    uncertainty_breakdown_{fam}_{vn}.png")


def shape_cov(cov, n):
    """Shape-only part of a covariance: remove the common normalisation,
    i.e. the covariance of h - n*sum(h)/sum(n)."""
    n = np.asarray(n, float)
    P = np.eye(len(n)) - np.outer(n / max(n.sum(), 1e-30), np.ones(len(n)))
    return P @ cov @ P.T


def _corr(cov):
    dg = np.sqrt(np.diag(cov))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.outer(dg, dg) > 1e-20, cov / np.outer(dg, dg), np.nan)


def _correlations(vn):
    n_bins = len(BINNING[vn]) - 1; tl = C.bin_labels(vn)
    panels = [(flags.cov_source, f'Total ({flags.cov_source})')] + \
             [(f, FAMILY_LABEL[f]) for f in BUDGET_FAMILIES]
    panels = [(n, l, load_cov(n, vn)) for n, l in panels]
    panels = [p for p in panels if p[2] is not None]
    if not panels:
        return
    for kind in ('', '_shape'):
        ncol = 4; nrow = -(-len(panels) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.2 * nrow), squeeze=False)
        for k, (n, l, d) in enumerate(panels):
            ax = axes[k // ncol][k % ncol]
            cov = shape_cov(d['cov'], d['mean_hist']) if kind else d['cov']
            im = ax.imshow(np.ma.masked_invalid(_corr(cov)), origin='lower', vmin=-1, vmax=1, cmap='RdBu_r')
            ax.set_title(l, fontsize=11)
            ax.set_xticks(range(n_bins)); ax.set_xticklabels(tl, fontsize=7, rotation=45, ha='right')
            ax.set_yticks(range(n_bins)); ax.set_yticklabels(tl, fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for k in range(len(panels), nrow * ncol):
            axes[k // ncol][k % ncol].set_visible(False)
        plt.suptitle(f'Correlation matrices by family: {vn}' +
                     ('  (shape only: common normalisation removed)' if kind else ''))
        plt.tight_layout()
        plt.savefig(f'{flags.plot_dir}/correlation_by_source{kind}_{vn}.png'); plt.close()
        print(f"    correlation_by_source{kind}_{vn}.png")


def _snapshots(vals, mc_w, injected, push_files):
    n = len(push_files); snap = sorted({s for s in (1, 3, 5, n) if s <= n})
    cols = plt.cm.viridis(np.linspace(0.2, 0.9, len(snap)))
    for vn in C.VARS:
        b = BINNING[vn]; cen = 0.5 * (b[:-1] + b[1:])
        prior = C.hist(vn, vals[vn], mc_w); truth = C.hist(vn, vals[vn], mc_w * injected)
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        step(axes[0], vn, prior, color='gray', linewidth=2, linestyle='--', label='Prior')
        step(axes[0], vn, truth, color='black', linewidth=2.5, label='Tilted data')
        for c, s in zip(cols, snap):
            h = C.hist(vn, vals[vn], mc_w * C.load_push(push_files[s - 1]))
            axes[0].errorbar(cen, h, yerr=np.sqrt(np.maximum(h, 0)), fmt='o', color=c, label=f'Iter {s}')
            axes[1].plot(cen, h / np.where(truth > 0, truth, 1), 'o-', color=c)
        axes[1].plot(cen, prior / np.where(truth > 0, truth, 1), 's--', color='gray')
        axes[1].axhline(1, color='black'); axes[1].set_ylim(0.7, 1.3); axes[1].set_ylabel('Ratio to truth')
        axes[1].set_xlabel(xlabel(vn)); set_bin_ticks(axes[1], vn)
        axes[0].legend(fontsize=9, ncol=2); axes[0].set_ylabel('Weighted events')
        axes[0].set_title(f'Reweighting snapshots: {vn} ({TAG})', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{flags.plot_dir}/reweighting_snapshots_{TAG}_{vn}.png'); plt.close()


def _weights(vals, mc_w, injected, push, push_files):
    # per-event weight profile
    for vn in C.VARS:
        b = BINNING[vn]; cen = 0.5 * (b[:-1] + b[1:]); x = C.fold(vn, vals[vn])
        fig, (axs, ax) = plt.subplots(1, 2, figsize=(15, 5.5))
        order = np.argsort(injected)
        xr = np.clip(np.asarray(vals[vn], float), b[0], b[-1] - 1e-6 * (b[-1] - b[0]))
        lim = max(abs(injected.max() - 1), abs(1 - injected.min()))
        sc = axs.scatter(xr[order], push[order], c=injected[order], s=4, alpha=0.6,
                         cmap='coolwarm', vmin=1 - lim, vmax=1 + lim)
        axs.axhline(1, color='black', linestyle='--', linewidth=1)
        axs.set_ylim(0, 3); axs.set_ylabel('Per-event push weight')
        set_bin_ticks(axs, vn); axs.set_xlabel(xlabel(vn))
        axs.set_title(f'Per-event weights: {vn} ({TAG})', fontsize=12)
        plt.colorbar(sc, ax=axs, label='Injected weight')
        mean = np.full(len(cen), np.nan); sd = mean.copy(); inj = mean.copy()
        for i in range(len(cen)):
            m = (x >= b[i]) & (x < b[i+1])
            if m.any():
                mean[i], sd[i], inj[i] = push[m].mean(), push[m].std(), injected[m].mean()
        ax.errorbar(cen, mean, yerr=sd, fmt='ro-', capsize=4, label=r'Push mean $\pm$ std')
        ax.plot(cen, inj, 'g^-', label='Injected tilt')
        ax.axhline(1, color='black', linestyle='--'); ax.set_ylim(0, 3)
        ax.set_xlabel(xlabel(vn)); ax.set_ylabel('Weight'); set_bin_ticks(ax, vn); ax.legend()
        ax.set_title(f'Weight profile: {vn} ({TAG})', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{flags.plot_dir}/weights_vs_observable_{TAG}_{vn}.png'); plt.close()
    # distributions
    n = len(push_files); snap = sorted({s for s in (1, 3, 5, n) if s <= n})
    fig, ax = plt.subplots(figsize=(6, 5))
    for c, s in zip(plt.cm.viridis(np.linspace(0.2, 0.9, len(snap))), snap):
        ax.hist(C.load_push(push_files[s - 1]), bins=100, range=(0.5, 2), density=True,
                alpha=0.5, color=c, label=f'Iter {s}')
    ax.axvline(1, color='black', linestyle='--'); ax.legend()
    ax.set_xlabel('Push weight'); ax.set_ylabel('Density'); ax.set_title(f'Push weights ({TAG})')
    plt.tight_layout(); plt.savefig(f'{flags.plot_dir}/weight_distributions_{TAG}.png'); plt.close()
    # 2D map
    kb, cb = BINNING[C.EVAR], BINNING['true_costheta']
    ke, cs = C.fold(C.EVAR, vals[C.EVAR]), vals['true_costheta']
    s, _, _ = np.histogram2d(ke, cs, bins=[kb, cb], weights=push)
    c, _, _ = np.histogram2d(ke, cs, bins=[kb, cb])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(np.arange(len(cb)), np.arange(len(kb)), np.where(c > 0, s / np.maximum(c, 1), np.nan),
                       cmap='RdBu_r', vmin=0.8, vmax=1.2)
    ax.set_xticks(np.arange(len(cb))); ax.set_xticklabels([f'{v:g}' for v in cb])
    ax.set_yticks(np.arange(len(kb))); ax.set_yticklabels([f'{v:g}' for v in kb[:-1]] + ['∞'])
    ax.set_xlabel(XLABEL['true_costheta']); ax.set_ylabel(XLABEL[C.EVAR])
    ax.set_title(f'Mean push weight ({TAG})'); plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(f'{flags.plot_dir}/weight_map_2d_{TAG}.png'); plt.close()


def _cov_2d(vals, mc_w, man, kb, cb):
    """Total covariance of the 2D (cosθ-major, KE-minor) histogram for the families in
    --cov-source, built from the same universes as the 1D covariances."""
    ke, cs = C.fold(C.EVAR, vals[C.EVAR]), vals['true_costheta']
    n2 = (len(kb) - 1) * (len(cb) - 1)
    total = np.zeros((n2, n2)); n_src = 0
    fams = C.families_in(flags.cov_source) or C.FAMILIES
    for key, m in man.items():
        if m['family'] not in fams:
            continue
        try:
            _, arrs = C.universe_weight_arrays(key, flags.mode, mc_w, man, flags.export_dir,
                                               flags.weights_base)
        except (ValueError, FileNotFoundError) as e:
            print(f"    2D cov: skip {key}: {e}"); continue
        if len(arrs) < 2:
            continue
        hs = [np.histogram2d(cs, ke, bins=[cb, kb], weights=mc_w * a)[0].ravel() for a in arrs]
        total += C.covariance_from_hists(hs)[0]; n_src += 1
    nom = np.histogram2d(cs, ke, bins=[cb, kb], weights=mc_w)[0].ravel()
    return total, nom, n_src


def _slices_2d(vals, mc_w, tilt, push, cov2d):
    """d2σ/dT dcosθ in slices of each variable: points = stat, band = total unc."""
    cov, _, n_src = cov2d
    nk, nc = len(BINNING[C.EVAR]) - 1, len(BINNING['true_costheta']) - 1
    for outer, inner in (('true_costheta', C.EVAR), (C.EVAR, 'true_costheta')):
        ob, ib = BINNING[outer], BINNING[inner]
        xo, xi = C.fold(outer, vals[outer]), C.fold(inner, vals[inner])
        n_sl = len(ob) - 1; ncol = 3; nrow = -(-n_sl // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.3 * ncol, 4.4 * nrow), squeeze=False)
        cen = 0.5 * (ib[:-1] + ib[1:]); lab = C.bin_labels(outer)
        for s in range(nrow * ncol):
            ax = axes[s // ncol][s % ncol]
            if s >= n_sl:
                ax.set_visible(False); continue
            m = (xo >= ob[s]) & (xo < ob[s + 1])
            sc = np.diff(ib) * (ob[s + 1] - ob[s])
            th, _ = np.histogram(xi[m], bins=ib, weights=(mc_w * tilt)[m])
            nh, _ = np.histogram(xi[m], bins=ib, weights=mc_w[m])
            uh, _ = np.histogram(xi[m], bins=ib, weights=(mc_w * push)[m])
            u2, _ = np.histogram(xi[m], bins=ib, weights=((mc_w * push) ** 2)[m])
            g = u2 > 0
            c2 = float(np.sum((uh[g] - th[g]) ** 2 / u2[g])); ndf = max(int(g.sum()) - 1, 1)
            if n_src:
                idx = ([s * nk + i for i in range(nk)] if outer == 'true_costheta'
                       else [i * nk + s for i in range(nc)])
                rel = np.sqrt(np.diag(cov)[idx]) / np.maximum(nh, 1e-12)
                _band(ax, inner, uh / sc, rel * uh / sc, color='red', alpha=0.15, linewidth=0,
                      label=f'Total unc. ({flags.cov_source})')
            step(ax, inner, th / sc, color='black', linewidth=2, label='Tilted data')
            step(ax, inner, nh / sc, color='gray', linestyle='--', label='Prior')
            ax.errorbar(cen, uh / sc, yerr=np.sqrt(u2) / sc, fmt='o', color='red', markersize=4,
                        label='OmniFold (stat.)')
            ax.set_title(f'{outer} ∈ {lab[s]}', fontsize=11)
            set_bin_ticks(ax, inner); ax.set_xlabel(XLABEL[inner], fontsize=10)
            ax.tick_params(axis='x', labelsize=8)
            ax.set_ylim(bottom=0)
            legend_with_box(ax, fit_annotation(c2, ndf, '[stat]'), LEGEND_SIDE[inner],
                            fontsize=7, headroom=1.9)
        plt.suptitle(rf'SBND $\nu_e$ CC $d^2\sigma$ in slices of {outer} ({TAG})')
        plt.tight_layout()
        plt.savefig(f'{flags.plot_dir}/xsec_2d_slices_{outer}_{TAG}.png'); plt.close()
        print(f"    xsec_2d_slices_{outer}_{TAG}.png")


def _correlation_2d(vals, mc_w, man):
    """KE x cosθ correlation in the coarser C.CORR2D_BINS (readable labels)."""
    kb, cb = C.CORR2D_BINS[C.EVAR], C.CORR2D_BINS['true_costheta']
    cov, nom, n_src = _cov_2d(vals, mc_w, man, kb, cb)
    if n_src == 0:
        print("    Not enough universes for the 2D correlation"); return
    nk, nc = len(kb) - 1, len(cb) - 1; n2 = nk * nc
    kl = [f'{kb[i]:g}-{kb[i+1]:g}' if i < nk - 1 else f'{kb[i]:g}+' for i in range(nk)]
    cl = [f'{cb[i]:g}-{cb[i+1]:g}' for i in range(nc)]
    e = 'p' if C.EVAR == 'true_p' else 'KE'
    labels = [f'cos {cl[ic]}\n{e} {kl[ik]}' for ic in range(nc) for ik in range(nk)]
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, c, ttl in ((axes[0], cov, 'total'), (axes[1], shape_cov(cov, nom), 'shape only')):
        im = ax.pcolormesh(np.arange(n2 + 1), np.arange(n2 + 1), np.ma.masked_invalid(_corr(c)),
                           vmin=-1, vmax=1, cmap='RdBu_r')
        for i in range(n2):
            for j in range(n2):
                v = _corr(c)[i, j]
                if np.isfinite(v):
                    ax.text(j + 0.5, i + 0.5, f'{v:.2f}', ha='center', va='center', fontsize=7,
                            color='white' if abs(v) > 0.6 else 'black')
        ax.set_xticks(np.arange(n2) + 0.5); ax.set_xticklabels(labels, fontsize=8, rotation=90)
        ax.set_yticks(np.arange(n2) + 0.5); ax.set_yticklabels(labels, fontsize=8)
        for ic in range(1, nc):
            ax.axhline(ic * nk, color='black', linewidth=1); ax.axvline(ic * nk, color='black', linewidth=1)
        ax.set_title(f'{ttl}  ({flags.cov_source}, {n_src} sources)', fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(rf'Correlation of the ({C.SHORT[C.EVAR]}, $\cos\theta_e$) cross section, {nk}$\times${nc} coarse bins')
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/correlation_2d_{C.TILT_SHORT[C.EVAR]}_costheta.png'); plt.close()
    print(f"    correlation_2d_{C.TILT_SHORT[C.EVAR]}_costheta.png ({n_src} sources)")


if flags.action in ('validation', 'all'):
    do_validation(flags.val_dir)
if flags.action in ('results', 'all'):
    do_results()
