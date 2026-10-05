# OmniFold for the SBND νe CC cross section

This repository unfolds selected νe CC events with OmniFold and computes the
systematic uncertainties from reweighted universes. The analysis variables are
the true electron momentum (`true_p`) and `cos(theta)`, and the sample is all
selected true signal events up to the start-dE/dx cut (5,357 events).

```
selection pickle (cafpyana) ──> FormatData_SBND.py ──> OmniFold inputs
notebook universe spectra   ──> ExportWeights.py   ──> per-event universe weights
                                                         │
          closure test, fake-data test, systematic universes (run_sbnd.py)
                                                         │
                      BuildResults.py (covariances, cross section)
                                                         │
                              MakePlots.py (all plots)
```

## Files

| file | what it does |
|---|---|
| `setup.sh` | activates the Python/TensorFlow environment (`--install` builds one) |
| `run_sbnd.py`, `omnifold.py`, `utils.py` | one OmniFold training; called by the scripts below |
| `sbnd/sbnd_config.py` | paths, binning, labels and systematic families, used by every script |
| `sbnd/FormatData_SBND.py` | selection pickle -> OmniFold inputs and efficiency |
| `sbnd/ExportWeights.py` | notebook universe spectra -> per-event universe weights (already done for the tutorial) |
| `sbnd/RunStudies.py` | list and screen the systematics, closure check, fake data, train universes, progress |
| `sbnd/runOmnifold_sbnd_closure.sh` | closure training |
| `sbnd/runOmnifold_sbnd_fakedata.sh` | fake-data training |
| `sbnd/launch_syst.sh` | parallel training of the systematic universes and ML replicas (already done for the tutorial) |
| `sbnd/BuildResults.py` | covariance matrices and cross section |
| `sbnd/MakePlots.py` | validation and result plots |

All python scripts list their options with `-h`, e.g. `python3 sbnd/RunStudies.py run-syst -h`.

## Inputs and outputs

The inputs are read from the shared data area and are never written to:

```
/exp/sbnd/data/users/castalyf/OmniFold_SBND_data/
├── selected_nuecc_signal.pkl     events from the cafpyana notebook, with truth, reco and the selection flags
├── exported_weights/             per-event weights, one file per systematic source (5357 events x 100 universes)
│   ├── <source>_universe_weights.npy
│   ├── mcstat_universe_weights.npy
│   └── universe_weights_manifest.json   list of sources and their family
└── pretrained_runs/              systematic universes and ML replicas already trained (final iteration only)
    ├── weights_<source>/<source>_univ<i>/
    └── weights_ml_unc/tilt_p_alpha0.3/replica_<i>/
```

Everything you produce stays inside your copy of the repository:
`FormattedData/`, `sbnd/runs/`, `sbnd/covariance/`, `sbnd/plots_*/` and `logs/`.
Paths are set in `sbnd/sbnd_config.py` and can be changed with environment
variables (listed at the top of that file). For the tutorial none are needed.

## Systematics

Every systematic **source** (one knob, 100 universes) gets its own covariance.
Sources are independent, so they are added into **families**:

| family | label on plots | content |
|---|---|---|
| `flux` | BNB Flux | 13 flux knobs |
| `genie` | GENIE XSec | 52 GENIE knobs |
| `extra_xsec` | Other XSec | cross-section uncertainties outside GENIE |
| `g4` | G4 Reint. | 3 Geant4 reinteraction knobs |
| `mcstat` | MC Stat | Poisson(1) weight per event |
| `ml` | ML | spread of 50 trainings that differ only in the network initialisation |

and the families into a **group** (used by `--source` and `--cov-source`):

| group | families |
|---|---|
| `all` | flux + genie + extra_xsec + g4 + mcstat (+ ml) |
| `fds` | mcstat + genie + extra_xsec (what the fake-data test is compared with) |
| `xsec` | genie + extra_xsec |
| `syst` | everything except mcstat |

There are two ways to get a universe's unfolded spectrum:

- **unfolded**: the universe weights are applied to the MC to make pseudo-data,
  and OmniFold unfolds it with the nominal MC (one training per universe). This
  includes how the unfolding responds to the change.
- **direct**: the universe weights are applied straight to the true spectrum,
  without training. This assumes the unfolding gets the change exactly right,
  which is fine for sources that are small or only change the normalisation.

`RunStudies.py screen` looks at the shape-only size of each source and picks
the ones worth training (above 0.5 % in any bin); these are the `screened`
sources. The covariance then uses `hybrid` mode: unfolded universes for the
screened sources and direct for the rest. Hybrid is the one we quote; `direct`
for everything is a quick cross-check.

Universes keep their normalisation (`KEEP_NORM = True` in `sbnd_config.py`), so
rate uncertainties are included, as needed for an absolute cross section.

---

# Tutorial

The systematic universes and ML replicas take many hours to train, so they are
taken from `pretrained_runs/`. The rest is run with smaller settings so it fits
in about 15 minutes. Run everything on EAF, where `/exp/sbnd` is visible.

## 0. Get the code (1 min)

```bash
cd ~
git clone https://github.com/castalyfan1012/OmniFold_SBND_tutorial.git
cd OmniFold_SBND_tutorial
source setup.sh
```

`setup.sh` activates the shared Python environment; it should print
`TensorFlow: 2.15.0`. Run all commands from this folder. In a new terminal,
`cd` here and `source setup.sh` again.

## 1. Format the inputs (30 s)

```bash
python3 sbnd/FormatData_SBND.py
```

Keeps the true signal events that pass the last selection cut
(`sel_start_dedx`) and writes the network inputs, the raw truth values and the
selection efficiency to `FormattedData/`. It should say
`Final N (reco & truth): 5,357`.

`--sel-file` and `--final-stage` choose another pickle or another cut.

## 2. Closure test, in the background (2-3 min)

```bash
nohup bash sbnd/runOmnifold_sbnd_closure.sh --niter 3 --ntrial 1 &
```

The nominal MC is unfolded onto itself, so all weights should come out close to 1.

- `--niter`: OmniFold iterations (default 10)
- `--ntrial`: networks averaged in each iteration (default 7). More networks means less noise and a longer run.

`tail -f logs/closure.log` follows the training (Ctrl-C stops `tail` only).

## 3. Systematic sources (1 min)

```bash
python3 sbnd/RunStudies.py list-sources
python3 sbnd/RunStudies.py screen
```

`list-sources` prints every source, its family and how many universes have been
trained so far.

`screen` computes each source without any training: the size per family and
bin, with and without normalisation, and which sources should be trained
(saved to `sbnd/covariance/screen_sources.json`). It also compares the sum of
the single GENIE and flux knobs with the combined throws.

- `--thresh 0.005`: train a source if its shape-only uncertainty is above 0.5 % in any bin
- `--shape-only`: rescale every universe to the nominal total (default is to keep the normalisation)

## 4. Fake-data test (3-5 min)

```bash
python3 sbnd/RunStudies.py make-fakedata --var true_p --alpha 0.3
nohup bash sbnd/runOmnifold_sbnd_fakedata.sh --var true_p --alpha 0.3 --niter 5 --ntrial 1 &
```

The fake data is the MC with its truth reweighted by `1 + alpha * z`, where `z`
is the standardised momentum. If the unfolding works it gives back this tilted
truth.

- `--var`: which truth variable is tilted. `true_p` (used here), `true_costheta`, or `both` to tilt momentum and angle together.
- `--alpha`: tilt strength (0.3 is 30 % per standard deviation)
- `--mode universe --source <name> --universe-idx <i>`: use one systematic universe as fake data instead of a tilt

The sample is labelled by a tag, here `tilt_p_alpha0.3` (`tilt_costheta_...`,
`tilt_both_...` for the others). The training goes to
`sbnd/runs/weights_sbnd_fakedata_tilt_p_alpha0.3/` and its log to
`logs/fdt_tilt_p_alpha0.3.log`. The injected spectrum is plotted in
`sbnd/plots_validation/fakedata_injected_tilt_p_alpha0.3.png`.

## 5. Pretrained systematics (30 s)

```bash
mkdir -p sbnd/runs
ln -s /exp/sbnd/data/users/castalyf/OmniFold_SBND_data/pretrained_runs/weights_* sbnd/runs/
python3 sbnd/RunStudies.py status
```

`status` shows how many universes are done for each source and how many ML
replicas there are for each fake-data tag. In the full analysis they are
trained with (not needed now):

```bash
python3 sbnd/RunStudies.py run-syst --source screened --dry-run   # list what would be trained
bash sbnd/launch_syst.sh 3 2 --with-ml                            # 3 workers x 2 CPUs + 50 ML replicas
```

- `launch_syst.sh N T --with-ml`: N workers with T CPUs each (keep N x T within `nproc`); `--with-ml` also trains the ML replicas
- `run-syst --source`: `screened`, a group, a family, one source, or a comma-separated list
- `run-syst --worker i --n-workers N`: worker i trains every N-th universe
- `run-syst --niter --ntrial --epochs`: training settings (10, 7, 100)
- `run-syst --redo`: retrain universes that are already done; without it they are skipped, so relaunching is safe
- `run-ml-unc --n-replicas 50 --var true_p --alpha 0.3`: trains the same fake data 50 times with one network each; the spread is the ML uncertainty

## 6. Closure check (30 s)

Once `logs/closure.log` ends with `Done`:

```bash
python3 sbnd/RunStudies.py check-closure
```

Compares the unfolded spectrum with the nominal truth using MC-stat errors and
passes if the p-value is above 0.05 (`--pval-thresh`). Plots:
`sbnd/plots_validation/closure_*.png`.

## 7. Covariances (1-2 min)

```bash
python3 sbnd/BuildResults.py covariance --source ml --ml-tag tilt_p_alpha0.3 --ml-label 50rep
python3 sbnd/BuildResults.py covariance --source all --freeze-ml --ml-as-stderr
python3 sbnd/BuildResults.py covariance --source fds
```

The first line reads the 50 ML replicas and saves their covariance.
- `--ml-tag`: the fake data the replicas were trained on
- `--ml-label 50rep`: also save a copy called `covariance_ml_50rep_<var>.npz`, so it is kept if you later build one from a different number of replicas

The second line builds the total of all families.
- `--freeze-ml`: use the ML covariance saved by the first line instead of reading the replicas again
- `--ml-as-stderr`: divide the ML covariance by the number of replicas. The central value we quote is the mean of the 50 replicas, and the uncertainty on a mean of N is σ/√N. Without this flag the ML term is the spread of one single training.
- `--no-ml`: leave the ML term out

The third line builds the `fds` group used in the fake-data plots.

The printout lists every source (with `unfolded` or `direct`) and the
breakdown by family per bin. Other options:
- `--mode hybrid` (default), `unfolded` (trained universes only) or `direct` (no training at all). They write the same files, so rerun hybrid after trying the others.
- `--source genie` or `--source genie__MaCCRES`: one family or one source
- `--var true_p`: one variable only (default: both)

## 8. Cross section (30 s)

```bash
python3 sbnd/BuildResults.py xsec --tilted-var true_p --alpha 0.3 --cov-source all
```

Cross section from the fake-data training, with statistical error bars and the
total uncertainty of `--cov-source` as a band
(`sbnd/plots_xsec/xsec_fdt_tilt_p_alpha0.3_<var>.png`), plus a table in
`sbnd/covariance/`.

- `--tilted-var`, `--alpha` (or `--tag`): which fake data
- `--cov-source`: group shown as the band (`all`, `fds`, `xsec`, `syst`)

## 9. Plots (1-2 min)

```bash
python3 sbnd/MakePlots.py all --var true_p --alpha 0.3
```

- `sbnd/plots_validation/`: weight recovery, unfolded vs injected spectrum, χ²/ndf vs iteration, and a pass/fail line in the printout
- `sbnd/plots_xsec/`: mean of the ML replicas vs the injected truth (`xsec_ml_*`), correlation matrices, 2D slices and the 2D correlation
- `sbnd/plots_syst/`: uncertainty budget by family, with and without normalisation, and the largest sources in each family

Options:
- `--iter`: OmniFold iteration to use (default 10, or the last one trained)
- `--cov-source`: band in the result plots (default `all`); `--fds-cov-source`: band in the validation plots (default `fds`)
- `--top-n 8`: sources shown per family breakdown
- `validation` or `results` instead of `all` runs one part only

## Things to try

- Rerun the closure with the full settings (`--niter 10 --ntrial 7`) and compare.
- Make fake data with `--var true_costheta` or `--var both`, train it, and see whether it is recovered. There are no ML replicas for these tags, so `xsec_ml_*` is skipped.
- Run `covariance --source all --mode direct` and compare the totals with hybrid.
- In `uncertainty_budget_true_p.png`, which family dominates, and how much of it is normalisation?
