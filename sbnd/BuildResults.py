"""
Covariance matrices and cross sections from the OmniFold trainings.

    python3 sbnd/BuildResults.py covariance [options]
    python3 sbnd/BuildResults.py xsec [options]

covariance
    For every systematic source: the covariance of the unfolded spectrum over
    its universes. Sources are independent, so they are summed into a family
    covariance (flux, genie, extra_xsec, g4, mcstat) and into a total for the
    chosen group. Written to sbnd/covariance/ as
        covariance_<source>_<var>.npz, covariance_<family>_<var>.npz,
        covariance_<group>_<var>.npz

    --source    all   flux + genie + extra_xsec + g4 + mcstat (+ ML)
                fds   mcstat + genie + extra_xsec (stat and cross section only)
                xsec  genie + extra_xsec
                syst  everything except MC stat
                ml    only the ML replicas
                or one family / source name
    --mode      hybrid    trained universes where they exist, untrained otherwise (default)
                unfolded  trained universes only
                direct    no training at all: universe weights applied to the truth MC

xsec
    Cross section of the single fake-data training with statistical error bars
    and the total uncertainty of --cov-source as a band; also saved as a table
    (sbnd/covariance/xsec_fdt_<tag>_<var>.csv).

Examples:
    python3 sbnd/BuildResults.py covariance --source ml --ml-tag tilt_p_alpha0.3
    python3 sbnd/BuildResults.py covariance --source all --freeze-ml --ml-as-stderr
    python3 sbnd/BuildResults.py covariance --source all --mode direct
    python3 sbnd/BuildResults.py xsec --tilted-var true_p --alpha 0.3 --cov-source all
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sbnd_config as C
from sbnd_config import BINNING, XLABEL, YLABEL_XSEC, chi2_pvalue, fit_annotation, make_fdt_tag

parser = argparse.ArgumentParser(description='Covariances and cross sections (see the top of this file).')
sub = parser.add_subparsers(dest='action', required=True)

p_cov = sub.add_parser('covariance', help='build covariance matrices')
p_cov.add_argument('--source', default='all',
                   help='group (all, fds, xsec, syst), family, source name, comma list, or ml')
p_cov.add_argument('--mode', choices=['hybrid', 'unfolded', 'direct'], default='hybrid',
                   help='use trained universes (unfolded), none (direct) or both (hybrid)')
p_cov.add_argument('--keep-norm', dest='keep_norm', action='store_true',
                   help='universes keep their normalisation (default)')
p_cov.add_argument('--shape-only', dest='keep_norm', action='store_false',
                   help='rescale universes to the nominal total (must match run-syst)')
p_cov.set_defaults(keep_norm=C.KEEP_NORM)
p_cov.add_argument('--var', choices=C.VARS + ['both'], default='both')
p_cov.add_argument('--ml-tag', default=C.make_fdt_tag(C.EVAR, 0.3),
                   help='fake data the ML replicas were trained on (default tilt_p_alpha0.3)')
p_cov.add_argument('--ml-label', default=None,
                   help='also save the ML covariance as covariance_ml_<label>_<var>.npz')
p_cov.add_argument('--freeze-ml', action='store_true',
                   help='reuse the saved ML covariance instead of reading the replicas again')
p_cov.add_argument('--ml-as-stderr', action='store_true',
                   help='divide the ML covariance by the number of replicas '
                        '(right when the result is the mean of the replicas)')
p_cov.add_argument('--no-ml', action='store_true', help='leave the ML covariance out of --source all')
p_cov.add_argument('--stability', action='store_true',
                   help='also print the chi2 of the universe mean vs nominal per iteration')
p_cov.add_argument('--data-dir', default=C.DATA_DIR)
p_cov.add_argument('--export-dir', default=C.EXPORT_DIR)
p_cov.add_argument('--weights-base', default=C.WEIGHTS_BASE)
p_cov.add_argument('--ml-weights-dir', default=None, help='default: $SBND_RUNS_DIR/weights_ml_unc/<ml-tag>/')
p_cov.add_argument('--cov-dir', default=C.COV_DIR)

p_xs = sub.add_parser('xsec', help='cross section of the single fake-data training')
p_xs.add_argument('--tilted-var', choices=C.TILT_CHOICES, default=C.EVAR,
                  help='which fake data (tilted variable)')
p_xs.add_argument('--alpha', type=float, default=0.3, help='which fake data (tilt strength)')
p_xs.add_argument('--tag', default=None, help='fake-data tag (instead of --tilted-var/--alpha)')
p_xs.add_argument('--cov-source', default='all', help='covariance used for the band (all, fds, ...)')
p_xs.add_argument('--var', choices=C.VARS + ['both'], default='both')
p_xs.add_argument('--data-dir', default=C.DATA_DIR)
p_xs.add_argument('--export-dir', default=C.EXPORT_DIR)
p_xs.add_argument('--plot-dir', default='sbnd/plots_xsec')
p_xs.add_argument('--cov-dir', default=C.COV_DIR)

flags = parser.parse_args()
if getattr(flags, 'ml_weights_dir', 'x') is None:
    flags.ml_weights_dir = C.ml_dir(flags.ml_tag)


# covariance
def _save(path, cov, mean_hist, nom_hist, bins, **extra):
    np.savez(path, cov=cov, frac_cov=C.frac_matrix(cov, mean_hist), bins=bins,
             mean_hist=mean_hist, nom_hist=nom_hist, **extra)


def _source_cov(key, var_name, vals, mc_w, man):
    """Covariance of one source, or None if it has fewer than 2 universes."""
    mode, arrs = C.universe_weight_arrays(
        key, flags.mode, mc_w, man, flags.export_dir, flags.weights_base, flags.keep_norm)
    if len(arrs) < 2:
        return None
    hs = np.array([C.hist(var_name, vals, mc_w * a) for a in arrs])
    cov, mean = C.covariance_from_hists(hs)
    return cov, mean, hs, mode


def _ml_cov(var_name, vals, mc_w, nom_hist, bins):
    path = f'{flags.cov_dir}/covariance_ml_{var_name}.npz'
    if flags.freeze_ml and os.path.exists(path):
        d = np.load(path)
        print(f"  using the saved ML covariance {path}")
        return d['cov'], d['mean_hist'], int(d['n_universes'])
    hs, stale = [], 0
    for rdir in sorted(glob.glob(flags.ml_weights_dir + 'replica_*/')):
        pf = C.pick_iter(glob.glob(rdir + 'Step2_Iter*_PushWeights.npy'))
        if not pf:
            continue
        push = C.load_push(pf)
        if len(push) != len(mc_w):
            stale += 1; continue
        hs.append(C.hist(var_name, vals, mc_w * push))
    if stale:
        print(f"  WARNING: {stale} ML replicas skipped (stale event count), rerun run-ml-unc")
    if len(hs) < 2:
        return None
    hs = np.array(hs)
    cov, mean = C.covariance_from_hists(hs)
    payload = dict(all_hists=hs, n_universes=np.array(len(hs)))
    _save(path, cov, mean, nom_hist, bins, **payload)
    label = flags.ml_label or f'{len(hs)}rep'
    _save(f'{flags.cov_dir}/covariance_ml_{label}_{var_name}.npz', cov, mean, nom_hist, bins,
          **payload)
    print(f"  ML: {len(hs)} replicas, saved covariance_ml_{var_name}.npz and _{label}")
    return cov, mean, len(hs)


def build_covariance_single_var(var_name):
    bins = BINNING[var_name]; n_bins = len(bins) - 1
    vals_all, mc_w = C.load_truth(flags.data_dir)
    vals = vals_all[var_name]
    nom_hist = C.hist(var_name, vals, mc_w)
    os.makedirs(flags.cov_dir, exist_ok=True)
    print(f"\nCovariance for {var_name} (source {flags.source}, mode {flags.mode})")

    # MC statistics from the summed squared weights (diagonal)
    stat_cov = np.diag(C.hist(var_name, vals, mc_w ** 2))
    _save(f'{flags.cov_dir}/covariance_stat_{var_name}.npz', stat_cov, nom_hist, nom_hist,
          bins, all_hists=nom_hist[None, :], n_universes=np.array(1))
    if flags.source == 'stat':
        print("  analytic MC-stat covariance saved"); return

    if flags.source == 'ml':
        _ml_cov(var_name, vals, mc_w, nom_hist, bins); return

    man = C.load_manifest(flags.export_dir)
    keys = C.resolve_sources(flags.source, man)
    fams_requested = C.families_in(flags.source)

    # per source
    fam_cov, fam_members, fam_modes = {}, {}, {}
    print(f"  {'source':<34s} {'mode':<9s} {'n':>4s}  frac unc per bin [%]")
    for key in keys:
        res = _source_cov(key, var_name, vals, mc_w, man)
        if res is None:
            print(f"  {C.source_label(key):<34s} (no universes, skipped)"); continue
        cov, mean, hs, mode = res
        _save(f'{flags.cov_dir}/covariance_{key}_{var_name}.npz', cov, mean, nom_hist, bins,
              all_hists=hs, n_universes=np.array(len(hs)), mode=np.array(mode),
              family=np.array(man[key]['family']))
        frac = np.sqrt(np.diag(cov)) / np.maximum(mean, 1e-12)
        print(f"  {C.source_label(key):<34s} {mode:<9s} {len(hs):>4d}  "
              + ' '.join(f'{x*100:5.2f}' for x in frac))
        fam = man[key]['family']
        fam_cov[fam] = fam_cov.get(fam, 0) + cov
        fam_members.setdefault(fam, []).append(key)
        fam_modes.setdefault(fam, set()).add(mode)

    # per family
    for fam, cov in fam_cov.items():
        if fam in fams_requested:   # a single-key run must not overwrite the family file
            complete = len(fam_members[fam]) == sum(m['family'] == fam for m in man.values())
            _save(f'{flags.cov_dir}/covariance_{fam}_{var_name}.npz', cov, nom_hist, nom_hist,
                  bins, members=np.array(fam_members[fam]),
                  n_sources=np.array(len(fam_members[fam])),
                  modes=np.array(sorted(fam_modes[fam])), complete=np.array(complete))

    # group total
    is_group = flags.source in C.GROUP_MEMBERS
    if not is_group and len(fam_cov) <= 1:
        return
    total = np.zeros((n_bins, n_bins)); parts = []
    for fam, cov in fam_cov.items():
        total += cov; parts.append(fam)
    if flags.source == 'fds' and 'mcstat' not in fam_cov:
        total += stat_cov; parts.append('stat')
        print("  no MC-stat universes found, using the analytic MC-stat covariance")
    if flags.source == 'all' and not flags.no_ml:
        ml = _ml_cov(var_name, vals, mc_w, nom_hist, bins)
        if ml is not None:
            cml, _, n_u = ml
            total += cml / n_u if flags.ml_as_stderr else cml
            parts.append('ml')
    name = flags.source if is_group else '+'.join(parts)
    _save(f'{flags.cov_dir}/covariance_{name}_{var_name}.npz', total, nom_hist, nom_hist, bins,
          sources=np.array(parts), ml_as_stderr=np.array(flags.ml_as_stderr))

    # summary table
    labels = C.bin_labels(var_name)
    print(f"\n  Family breakdown (frac unc %), total saved as covariance_{name}_{var_name}.npz")
    hdr = ''.join(f'{l:>13s}' for l in labels)
    print(f"  {'':<14s}{hdr}")
    rows = [(C.FAMILY_LABEL[f], fam_cov[f]) for f in C.FAMILIES if f in fam_cov]
    if 'stat' in parts:
        rows.append((C.FAMILY_LABEL['stat'], stat_cov))
    rows.append(('TOTAL', total))
    for lab, cov in rows:
        f = np.sqrt(np.diag(cov)) / np.maximum(nom_hist, 1e-12)
        print(f"  {lab:<14s}" + ''.join(f'{x*100:13.2f}' for x in f))
    for fam in fam_cov:
        n_all = sum(m['family'] == fam for m in man.values())
        print(f"  {C.FAMILY_LABEL[fam]:<12s}: {len(fam_members[fam])}/{n_all} sources, "
              f"modes={sorted(fam_modes[fam])}")

    if flags.stability:
        _stability(var_name, vals, mc_w, nom_hist, keys)


def _stability(var_name, vals, mc_w, nom_hist, keys):
    """chi2 of the universe mean against the nominal spectrum at each iteration."""
    n_bins = len(BINNING[var_name]) - 1
    dirs = []
    for key in keys:
        dirs += sorted(glob.glob(f'{flags.weights_base}/weights_{key}/{key}_univ*/'))
    if not dirs:
        print("  (no trained universes for the stability table)"); return
    max_iter = max((C.iter_num(f) for f in glob.glob(dirs[0] + 'Step2_Iter*_PushWeights.npy')),
                   default=-1)
    print(f"\n  Systematic stability chi2 ({var_name}, {len(dirs)} universes):")
    for it in range(max_iter + 1):
        hs = []
        for d in dirs:
            pf = glob.glob(d + f'Step2_Iter{it}_*_PushWeights.npy')
            if pf:
                w = C.load_push(pf[0])
                if len(w) == len(mc_w):
                    hs.append(C.hist(var_name, vals, mc_w * w))
        if len(hs) < 2:
            continue
        cov, mu = C.covariance_from_hists(hs)
        cov += np.eye(n_bins) * 1e-6 * np.diag(cov).mean()
        c2 = float((mu - nom_hist) @ np.linalg.pinv(cov) @ (mu - nom_hist))
        print(f"    iter {it+1:3d}: chi2 = {c2:8.2f}, p = {chi2_pvalue(c2, n_bins - 1):.3f}")


# xsec
INTEGRATED_FLUX_PER_POT = 1.0
TARGET_POT              = 6.6e20
N_TARGETS               = 1.0


def extract_xsec_single_var(var_name):
    USE_ABSOLUTE = (INTEGRATED_FLUX_PER_POT != 1.0 and N_TARGETS != 1.0)
    NORM = INTEGRATED_FLUX_PER_POT * TARGET_POT * N_TARGETS if USE_ABSOLUTE else 1.0
    bins = BINNING[var_name]; n_bins = len(bins) - 1
    bin_widths = np.diff(bins); centers = 0.5 * (bins[:-1] + bins[1:])
    vals_all, mc_weights = C.load_truth(flags.data_dir)
    var_vals = vals_all[var_name]
    os.makedirs(flags.plot_dir, exist_ok=True)
    xsec_tag = flags.tag or make_fdt_tag(flags.tilted_var, flags.alpha)
    print(f"\nCross section for {var_name} (covariance: {flags.cov_source})")

    eff = C.load_efficiency(var_name, flags.export_dir, flags.data_dir)
    scale = eff.clip(1e-6) * bin_widths * NORM
    N_nom = C.hist(var_name, var_vals, mc_weights)
    xsec_nom = N_nom / scale

    push_files = sorted(glob.glob(C.fakedata_dir(xsec_tag) + 'Step2_Iter*_PushWeights.npy'),
                        key=C.iter_num)
    has_unf, push, xsec_unf = False, None, xsec_nom
    if push_files:
        push = C.load_push(C.pick_iter(push_files))
        if len(push) != len(mc_weights):
            print(f"  ERROR: push has {len(push)} events, sample has {len(mc_weights)}. "
                  f"Retrain the fake-data OmniFold run.")
        else:
            has_unf = True
            xsec_unf = C.hist(var_name, var_vals, mc_weights * push) / scale

    tilt_file = flags.data_dir + f'truth_weights_sbnd_fakedata_{xsec_tag}.npy'
    has_truth = os.path.exists(tilt_file)
    if has_truth:
        tilt = np.load(tilt_file)
        xsec_truth = C.hist(var_name, var_vals, mc_weights * tilt) / scale

    cov_file = f'{flags.cov_dir}/covariance_{flags.cov_source}_{var_name}.npz'
    has_syst = os.path.exists(cov_file)
    if has_syst:
        cd = np.load(cov_file)
        if cd['cov'].shape[0] != n_bins:
            print(f"  ERROR: {cov_file} uses the old binning, rebuild it."); has_syst = False
    if has_syst:
        xsec_unc = np.sqrt(np.diag(cd['cov'] / np.outer(scale, scale)))
        xsec_mean = cd['mean_hist'] / scale
    else:
        print(f"  WARNING: {cov_file} not found, build it with 'covariance --source {flags.cov_source}'")
        xsec_unc = np.zeros(n_bins); xsec_mean = xsec_nom

    gof = None
    if has_unf and has_truth:
        stat = C.hist(var_name, var_vals, (mc_weights * push) ** 2) / scale ** 2
        c2 = float(np.sum((xsec_unf - xsec_truth) ** 2 / stat.clip(1e-30)))
        gof = fit_annotation(c2, n_bins - 1, '[stat cov]')
        print(f"  Xsec vs tilted truth (stat cov): {gof.replace(chr(10), ', ')}")

    stat = (np.sqrt(C.hist(var_name, var_vals, (mc_weights * push) ** 2)) / scale
            if has_unf else np.zeros(n_bins))
    frac_unc = xsec_unc / np.maximum(xsec_mean, 1e-30)
    side = 'right' if var_name == C.EVAR else 'left'

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax = axes[0]
    if has_syst and has_unf:
        for i in range(n_bins):
            ax.fill_between([bins[i], bins[i + 1]], xsec_unf[i] - xsec_unc[i], xsec_unf[i] + xsec_unc[i],
                            color='red', alpha=0.15, linewidth=0,
                            label=f'Total unc. ({flags.cov_source})' if i == 0 else None)
    if has_truth:
        ax.step(bins, np.append(xsec_truth, xsec_truth[-1]), where='post', color='black',
                linewidth=2, label='Tilted data')
    ax.step(bins, np.append(xsec_nom, xsec_nom[-1]), where='post', color='gray',
            linewidth=1.5, linestyle='--', label='Prior')
    if has_unf:
        ax.errorbar(centers, xsec_unf, yerr=stat, fmt='o', color='red', capsize=3,
                    label='OmniFold, single training (stat.)')
    ax.set_ylabel(YLABEL_XSEC[var_name]); ax.set_ylim(bottom=0)
    ax.set_title(rf'SBND $\nu_e$ CC Inclusive  ({xsec_tag}, single fake-data training)', fontsize=11)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi * 1.55)
    leg = ax.legend(loc=f'upper {side}', fontsize=9, framealpha=0.9)
    if gof:
        fig.canvas.draw()
        bb = leg.get_window_extent().transformed(ax.transAxes.inverted())
        ax.text(bb.x1 if side == 'right' else bb.x0, bb.y0 - 0.02, gof, transform=ax.transAxes,
                fontsize=9, va='top', ha=side, multialignment=side)
    if has_truth and has_unf:
        r = np.where(xsec_truth > 0, xsec_truth, 1)
        if has_syst:
            for i in range(n_bins):
                axes[1].fill_between([bins[i], bins[i + 1]], (xsec_unf[i] - xsec_unc[i]) / r[i],
                                     (xsec_unf[i] + xsec_unc[i]) / r[i], color='red', alpha=0.15, linewidth=0)
        axes[1].errorbar(centers, xsec_unf / r, yerr=stat / r, fmt='o', color='red', capsize=3)
    axes[1].axhline(1.0, color='black')
    axes[1].set_ylim(0.5, 1.5); axes[1].set_ylabel('OmniFold / Tilted')
    C.apply_axis(axes[1], var_name)
    plt.tight_layout()
    out = f'{flags.plot_dir}/xsec_fdt_{xsec_tag}_{var_name}.png'
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved {out}")

    labels = C.bin_labels(var_name)
    os.makedirs(flags.cov_dir, exist_ok=True)
    table = f'{flags.cov_dir}/xsec_fdt_{xsec_tag}_{var_name}.csv'
    with open(table, 'w') as fh:
        fh.write('bin_lo,bin_hi,efficiency,xsec_prior,xsec_omnifold,stat_unc,total_unc,frac_total_unc'
                 + (',xsec_tilted_truth' if has_truth else '') + '\n')
        for i in range(n_bins):
            hi_edge = 'inf' if (C.OVERFLOW[var_name] and i == n_bins - 1) else f'{bins[i+1]:g}'
            fh.write(f'{bins[i]:g},{hi_edge},{eff[i]:.5f},{xsec_nom[i]:.6e},{xsec_unf[i]:.6e},'
                     f'{stat[i]:.6e},{xsec_unc[i]:.6e},{frac_unc[i]:.5f}'
                     + (f',{xsec_truth[i]:.6e}' if has_truth else '') + '\n')
    print(f"\n  {'Bin':>16s} {'Eff':>7s} {'Prior':>12s} {'OmniFold':>12s} {'Stat':>11s} {'Total':>11s} {'Frac':>7s}")
    for i in range(n_bins):
        print(f"  {labels[i]:>16s} {eff[i]:7.3f} {xsec_nom[i]:12.4e} {xsec_unf[i]:12.4e} "
              f"{stat[i]:11.3e} {xsec_unc[i]:11.3e} {frac_unc[i]:7.3f}")
    print(f"  Table saved: {table}")


vars_to_run = C.VARS if flags.var == 'both' else [flags.var]
for v in vars_to_run:
    (build_covariance_single_var if flags.action == 'covariance' else extract_xsec_single_var)(v)
