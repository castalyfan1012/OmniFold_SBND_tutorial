"""
Settings and helpers shared by all the SBND OmniFold scripts: input/output
paths, binning, the list of systematic sources, and a few small utilities.

Inputs are read from the shared data area; everything produced goes inside
this repository. Any path can be changed with an environment variable:
    SBND_SHARED_DATA  shared data area (selection pickle, universe weights)
    SBND_SEL_FILE     selection pickle from the cafpyana notebook
    SBND_EXPORT_DIR   per-event systematic universe weights (+ manifest)
    SBND_DATA_DIR     where FormatData_SBND.py writes the OmniFold inputs
    SBND_RUNS_DIR     where all OmniFold trainings are written
    SBND_FINAL_STAGE  selection cut that defines the sample
    SBND_ENERGY_VAR   true_p (default) or true_ke
"""
import glob
import json
import os
import re

import numpy as np


def _dir(var, default):
    d = os.environ.get(var, default)
    return d if d.endswith('/') else d + '/'


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Inputs (read only)
SHARED_DATA = os.environ.get('SBND_SHARED_DATA',
                             '/exp/sbnd/data/users/castalyf/OmniFold_SBND_data').rstrip('/')
SEL_FILE    = os.environ.get('SBND_SEL_FILE', f'{SHARED_DATA}/selected_nuecc_signal.pkl')
EXPORT_DIR  = _dir('SBND_EXPORT_DIR', f'{SHARED_DATA}/exported_weights/')
FINAL_STAGE = os.environ.get('SBND_FINAL_STAGE', 'sel_start_dedx')
MANIFEST    = 'universe_weights_manifest.json'

# Outputs (all inside the repository)
DATA_DIR     = _dir('SBND_DATA_DIR', f'{REPO_DIR}/FormattedData/')
RUNS_DIR     = os.environ.get('SBND_RUNS_DIR', f'{REPO_DIR}/sbnd/runs').rstrip('/')
WEIGHTS_BASE = RUNS_DIR                        # <runs>/weights_<source>/<source>_univ<i>/
ML_DIR       = f'{RUNS_DIR}/weights_ml_unc/'   # <runs>/weights_ml_unc/<tag>/replica_<i>/
CONFIG_DIR   = f'{RUNS_DIR}/configs/'
FAILED_LOG   = f'{RUNS_DIR}/failed_universes.txt'
CLOSURE_DIR  = f'{RUNS_DIR}/weights_sbnd_closure/'
COV_DIR      = f'{REPO_DIR}/sbnd/covariance'


def ml_dir(tag):
    # ML replicas are trained on one fake-data sample, so they are kept per tag
    return f'{ML_DIR}{tag}/'


def fakedata_dir(tag):
    return f'{RUNS_DIR}/weights_sbnd_fakedata_{tag}/'


def closure_dir():
    return CLOSURE_DIR


# Unfolding settings used by every step
UNFOLD_ITER = 10     # OmniFold iteration used for all results

# KEEP_NORM = True: each systematic universe keeps its normalisation, so rate
# uncertainties are part of the covariance (what an absolute cross section needs).
# False rescales every universe to the nominal total (shape-only uncertainties).
# run-syst records the setting and refuses to mix the two.
KEEP_NORM = True


# Binning
# Columns of mc_vals_truth_NoNorm.npy (written by FormatData_SBND.py)
TRUTH_COL = {'true_ke': 0, 'true_costheta': 1, 'true_p': 2}

# OmniFold always trains on all three truth variables; EVAR only decides which
# energy-like variable the results are histogrammed in.
EVAR = os.environ.get('SBND_ENERGY_VAR', 'true_p')

BINNING = {
    EVAR:            np.array([500, 700, 1000, 1300, 1700, 2200, 3000], dtype=float),
    'true_costheta': np.array([-1.0, 0.6, 0.75, 0.85, 0.925, 1.0], dtype=float),
}
VARS = list(BINNING)

# coarser bins, only for the 2D correlation plot
CORR2D_BINS = {
    EVAR:            np.array([500, 1000, 1700, 3000], dtype=float),
    'true_costheta': np.array([-1.0, 0.75, 0.925, 1.0], dtype=float),
}

# events above the last energy edge go into the last bin
OVERFLOW = {EVAR: True, 'true_costheta': False}

XLABEL = {
    'true_p':        r'True electron momentum [MeV/c]',
    'true_ke':       r'True electron kinetic energy [MeV]',
    'true_costheta': r'True $\cos\theta_e$',
}
YLABEL_XSEC = {
    'true_p':        r'd$\sigma$/d$p_e$ [arb. / (MeV/c)]',
    'true_ke':       r'd$\sigma$/d$T_e$ [arb. / MeV]',
    'true_costheta': r'd$\sigma$/d$\cos\theta_e$ [arb.]',
}
SHORT = {'true_p': r'$p_e$', 'true_ke': 'KE', 'true_costheta': r'$\cos\theta_e$'}
BIN_FMT = {'true_p': '.0f', 'true_ke': '.0f', 'true_costheta': '.3g'}

# The first cos(theta) bin is very wide; on plots it is drawn narrower so the
# other tick labels have room. Only the drawing changes, not the contents.
DISPLAY_WIDTHS = {'true_costheta': np.array([0.2, 0.15, 0.10, 0.075, 0.075])}


def tick_labels(var_name):
    b = BINNING[var_name]
    lab = [f'{x:g}' for x in b]
    if OVERFLOW.get(var_name, False):
        lab[-1] = f'{b[-1]:g}+'
    return lab


def apply_axis(ax, var_name, axis='x', label=True, fontsize=None):
    """Bin edges as ticks, '3000+' on the overflow edge, narrower first cos(theta) bin."""
    from matplotlib.scale import FuncScale
    b = BINNING[var_name]
    if var_name in DISPLAY_WIDTHS:
        disp = np.concatenate([[0.0], np.cumsum(DISPLAY_WIDTHS[var_name])])
        fwd = lambda x: np.interp(x, b, disp)
        inv = lambda y: np.interp(y, disp, b)
        (ax.set_xscale if axis == 'x' else ax.set_yscale)(FuncScale(ax, (fwd, inv)))
    (ax.set_xticks if axis == 'x' else ax.set_yticks)(b)
    (ax.set_xticklabels if axis == 'x' else ax.set_yticklabels)(tick_labels(var_name),
                                                                fontsize=fontsize)
    (ax.set_xlim if axis == 'x' else ax.set_ylim)(b[0], b[-1])
    if label:
        (ax.set_xlabel if axis == 'x' else ax.set_ylabel)(XLABEL[var_name])


def apply_index_axis(ax, var_name, label=True):
    """Equal-width bins labelled with the bin edges. Returns the bin centres."""
    n = len(BINNING[var_name]) - 1
    ax.set_xticks(np.arange(n + 1))
    ax.set_xticklabels(tick_labels(var_name))
    ax.set_xlim(0, n)
    if label:
        ax.set_xlabel(XLABEL[var_name])
    return np.arange(n) + 0.5


def fold(var_name, vals):
    """Move overflow values into the last bin."""
    vals = np.asarray(vals, dtype=np.float64)
    bins = BINNING[var_name]
    if OVERFLOW.get(var_name, False):
        vals = np.where(vals >= bins[-1], 0.5 * (bins[-2] + bins[-1]), vals)
    return vals


def hist(var_name, vals, weights=None, bins=None):
    b = BINNING[var_name] if bins is None else bins
    h, _ = np.histogram(fold(var_name, vals), bins=b, weights=weights)
    return h


def bin_labels(var_name, bins=None):
    b = BINNING[var_name] if bins is None else bins
    f = BIN_FMT[var_name]
    lab = [f'[{b[i]:{f}},{b[i+1]:{f}})' for i in range(len(b) - 1)]
    if OVERFLOW.get(var_name, False) and bins is None:
        lab[-1] = f'[{b[-2]:{f}},∞)'
    return lab


def load_truth(data_dir=DATA_DIR):
    """Truth values of the analysis variables and the MC event weights."""
    truth_raw  = np.load(data_dir + 'mc_vals_truth_NoNorm.npy')
    mc_weights = np.load(data_dir + 'mc_weights_reco.npy')
    return {v: truth_raw[:, TRUTH_COL[v]] for v in VARS}, mc_weights


def load_efficiency(var_name, export_dir=EXPORT_DIR, data_dir=DATA_DIR):
    n_bins = len(BINNING[var_name]) - 1
    for d in (data_dir, export_dir):
        f = d + f'efficiency_{var_name}.npy'
        if os.path.exists(f):
            eff = np.load(f)
            if len(eff) != n_bins:
                raise ValueError(f"{f} has {len(eff)} bins, expected {n_bins}. "
                                 f"Rerun python3 sbnd/FormatData_SBND.py.")
            return eff
    print(f"  WARNING: no efficiency_{var_name}.npy found, using efficiency = 1")
    return np.ones(n_bins)


# Systematic sources
# Each entry of the manifest written by the notebook export is one source with
# 100 universes. Sources are grouped into families for the plots and summed
# (as independent sources) into family and total covariances.
FAMILIES = ['flux', 'genie', 'extra_xsec', 'g4', 'mcstat']

FAMILY_LABEL = {
    'flux': 'BNB Flux', 'genie': 'GENIE XSec', 'extra_xsec': 'Other XSec',
    'g4': 'G4 Reint.', 'mcstat': 'MC Stat', 'ml': 'ML/NN Init',
    'stat': 'MC Stat (analytic)',
}
FAMILY_COLOR = {
    'flux': 'tab:blue', 'genie': 'tab:red', 'extra_xsec': 'tab:orange',
    'g4': 'tab:brown', 'mcstat': 'tab:green', 'ml': 'tab:purple', 'stat': 'black',
}

LEGACY_FAMILY = {'bnb': 'flux'}

# Combined Flux / GENIE throws (all knobs varied together). Only used as a
# cross-check in `screen`, never added to the per-knob sources.
CROSSCHECK_FILES = {'flux': 'bnb_universe_weights.npy', 'genie': 'genie_universe_weights.npy'}

# Groups accepted by --source / --cov-source
GROUP_MEMBERS = {
    'all':  ['flux', 'genie', 'extra_xsec', 'g4', 'mcstat'],
    'fds':  ['mcstat', 'genie', 'extra_xsec'],      # stat + cross section
    'xsec': ['genie', 'extra_xsec'],
    'syst': ['flux', 'genie', 'extra_xsec', 'g4'],  # everything except MC stat
}


def load_manifest(export_dir=EXPORT_DIR):
    path = os.path.join(export_dir, MANIFEST)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Export the universe weights first.")
    with open(path) as fh:
        man = json.load(fh)
    for key, m in man.items():
        m.setdefault('family', key.split('__')[0] if '__' in key else key)
        m['family'] = LEGACY_FAMILY.get(m['family'], m['family'])
        m.setdefault('file', f'{key}_universe_weights.npy')
    return man


def family_of(key, manifest):
    if key in manifest:
        return manifest[key]['family']
    fam = key.split('__')[0] if '__' in key else key
    return LEGACY_FAMILY.get(fam, fam)


def source_label(key, manifest=None):
    """'genie__MaCCRES' -> 'GENIE: MaCCRES', 'extra_xsec__extra_xsec' -> 'Other XSec'"""
    if key in FAMILY_LABEL:
        return FAMILY_LABEL[key]
    fam, _, knob = key.partition('__')
    fam = LEGACY_FAMILY.get(fam, fam)
    if fam == 'extra_xsec':
        return 'Other XSec' if knob in ('', 'extra_xsec') else f'Other XSec: {knob}'
    if fam == 'g4':
        m = re.match(r'reinteractions_(\w+?)_Geant4', knob)
        return f'G4: {m.group(1)} reint.' if m else f'G4: {knob}'
    short = {'flux': 'Flux', 'genie': 'GENIE'}.get(fam, fam)
    return f'{short}: {knob}' if knob else FAMILY_LABEL.get(fam, key)


def resolve_sources(spec, manifest):
    """Turn a --source value into a list of sources. It can be a source name
    (genie__MaCCRES), a family (genie), a group (all, fds, xsec, syst) or a
    comma-separated mix of these."""
    out = []
    for tok in [t.strip() for t in spec.split(',') if t.strip()]:
        tok = LEGACY_FAMILY.get(tok, tok)
        if tok in manifest:
            out.append(tok)
        elif tok in GROUP_MEMBERS:
            for fam in GROUP_MEMBERS[tok]:
                out += [k for k, m in manifest.items() if m['family'] == fam]
        elif tok in FAMILIES:
            out += [k for k, m in manifest.items() if m['family'] == tok]
        else:
            raise KeyError(f"Unknown source '{tok}'. Use a source name, a family {FAMILIES} "
                           f"or a group {list(GROUP_MEMBERS)} "
                           f"(python3 sbnd/RunStudies.py list-sources shows the names).")
    seen = set()
    return [k for k in out if not (k in seen or seen.add(k))]


def families_in(spec):
    fams = []
    for tok in [t.strip() for t in spec.split(',') if t.strip()]:
        tok = LEGACY_FAMILY.get(tok, tok)
        fams += GROUP_MEMBERS.get(tok, [tok] if tok in FAMILIES else [])
    return list(dict.fromkeys(fams))


# Universe weights
def iter_num(p):
    m = re.search(r'Iter(\d+)', p)
    return int(m.group(1)) if m else -1


def load_push(path):
    w = np.load(path)
    return w if w.ndim == 1 else w.mean(axis=0)


def universe_dir(key, idx, weights_base=WEIGHTS_BASE):
    return f'{weights_base}/weights_{key}/{key}_univ{idx}/'


def unfolded_push_files(key, weights_base=WEIGHTS_BASE, n_iter=None):
    """Push-weight file of every finished universe of a source, {index: path}.
    A universe counts as finished only if it reached iteration UNFOLD_ITER."""
    n = UNFOLD_ITER if n_iter is None else n_iter
    out = {}
    for f in glob.glob(f'{weights_base}/weights_{key}/{key}_univ*/Step2_Iter{n-1}_*_PushWeights.npy'):
        m = re.search(r'_univ(\d+)/', f)
        if m:
            out[int(m.group(1))] = f
    return dict(sorted(out.items()))


def limit_threads(n_threads):
    """Environment for one training that uses at most n_threads CPUs."""
    env = os.environ.copy()
    for v in ('OMNIFOLD_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
        env[v] = str(n_threads)
    return env


def run_settings_path(key, weights_base=WEIGHTS_BASE):
    return f'{weights_base}/weights_{key}/run_settings.json'


def check_run_settings(key, keep_norm, weights_base=WEIGHTS_BASE):
    p = run_settings_path(key, weights_base)
    if os.path.exists(p):
        s = json.load(open(p))
        if bool(s.get('keep_norm')) != bool(keep_norm):
            print(f"  WARNING: {key} was trained with keep_norm={s.get('keep_norm')}, "
                  f"but this covariance uses keep_norm={keep_norm}.")


def data_weights_for_universe(mc_weights, uni_col, keep_norm=False):
    """Event weights that play the role of 'data' for one universe."""
    uni = np.clip(uni_col, 0.0, 10.0)
    dw = mc_weights * uni
    if not keep_norm:
        dw = dw * (mc_weights.sum() / dw.sum())
    return dw


def universe_weight_arrays(key, mode, mc_weights, manifest, export_dir=EXPORT_DIR,
                           weights_base=WEIGHTS_BASE, keep_norm=None, n_univ=None):
    """Per-event weights (relative to the nominal MC) for every universe of a source.

    mode 'unfolded': OmniFold push weights of the trained universes
    mode 'direct':   the exported universe weights applied to the truth MC (no training)
    mode 'hybrid':   'unfolded' if the source has trained universes, else 'direct'
    Returns (mode actually used, list of arrays).
    """
    n_ev = len(mc_weights)
    keep_norm = KEEP_NORM if keep_norm is None else keep_norm
    if mode in ('unfolded', 'hybrid'):
        files = unfolded_push_files(key, weights_base)
        if files:
            check_run_settings(key, keep_norm, weights_base)
        if n_univ is not None:
            files = {u: f for u, f in files.items() if u < n_univ}
        if files or mode == 'unfolded':
            arrs, stale = [], 0
            for f in files.values():
                w = load_push(f)
                if len(w) != n_ev:
                    stale += 1
                    continue
                arrs.append(w)
            if stale:
                print(f"  WARNING: {key}: {stale}/{len(files)} trained universes have a "
                      f"different number of events and were skipped (retrain them)")
            return 'unfolded', arrs
    info = manifest.get(key, {'file': f'{key}_universe_weights.npy'})
    path = os.path.join(export_dir, info['file'])
    uni_all = np.load(path, mmap_mode='r')
    if uni_all.shape[0] != n_ev:
        raise ValueError(f"{path} has {uni_all.shape[0]} rows but the OmniFold sample has "
                         f"{n_ev} events. The weights were exported for a different sample.")
    n = uni_all.shape[1] if n_univ is None else min(n_univ, uni_all.shape[1])
    arrs = [data_weights_for_universe(mc_weights, np.asarray(uni_all[:, i]), keep_norm)
            / mc_weights for i in range(n)]
    return 'direct', arrs


def covariance_from_hists(h):
    """Covariance of a set of universe histograms around their mean (1/N)."""
    h = np.asarray(h, dtype=np.float64)
    d = h - h.mean(axis=0)
    return (d.T @ d) / len(h), h.mean(axis=0)


def frac_matrix(cov, ref):
    den = np.outer(ref, ref)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, cov / den, 0.0)


# Misc
TILT_SHORT = {'true_p': 'p', 'true_ke': 'ke', 'true_costheta': 'costheta', 'both': 'both'}
TILT_CHOICES = [EVAR, 'true_costheta', 'both']


def make_fdt_tag(var, alpha):
    """Name of a fake-data sample, e.g. tilt_p_alpha0.3 or tilt_both_alpha0.3."""
    return f'tilt_{TILT_SHORT[var]}_alpha{alpha}'


def tilt_label(var, alpha):
    e = {'true_p': 'p', 'true_ke': 'KE'}[EVAR]
    names = {EVAR: f'{e} only', 'true_costheta': 'cosθ only', 'both': f'{e} + cosθ'}
    return f'Tilted: {names[var]}, α={alpha}'


def chi2_pvalue(chi2, ndf):
    if ndf is None or ndf <= 0 or not np.isfinite(chi2):
        return float('nan')
    try:
        from scipy import stats
        return float(stats.chi2.sf(chi2, ndf))
    except Exception:
        import math
        k, x = float(ndf), float(chi2)
        t = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
        return float(0.5 * math.erfc(t / math.sqrt(2.0)))


def fit_annotation(chi2, ndf, note=''):
    p = chi2_pvalue(chi2, ndf)
    return (rf'$\chi^2$/ndf = {chi2:.2f}/{ndf} = {chi2/ndf:.2f}' + '\n'
            + f'p = {p:.3f}' + (f'  {note}' if note else ''))


def pick_iter(files, n=None):
    """File of iteration n (1-based, default UNFOLD_ITER), or the last one available."""
    n = UNFOLD_ITER if n is None else n
    files = sorted(files, key=iter_num)
    for f in files:
        if iter_num(f) == n - 1:
            return f
    return files[-1] if files else None


def write_omnifold_config(path, cfg):
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=1)
