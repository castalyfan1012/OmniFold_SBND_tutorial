"""
Runs one OmniFold unfolding. Normally called by the shell scripts and by
RunStudies.py rather than by hand:

    python3 run_sbnd.py --config <config.json> --file_path <FormattedData dir> \
        --weights_folder <output dir> --no_eff --verbose

The push/pull weights of every iteration are saved in --weights_folder as
Step2_Iter<i>_<name>_PushWeights.npy and Step1_Iter<i>_<name>_PullWeights.npy.
"""

import os

# When RunStudies.py runs several trainings in parallel it sets OMNIFOLD_THREADS.
# Pinning each training to that many CPUs keeps TensorFlow from starting a thread
# pool per core, which otherwise hits the per-user process limit.
_NT = int(os.environ.get('OMNIFOLD_THREADS', '0') or 0)
if _NT > 0:
    try:
        _cpus = sorted(os.sched_getaffinity(0))
        _w = int(os.environ.get('OMNIFOLD_WORKER', os.getpid()))
        _start = (_w * _NT) % len(_cpus)
        os.sched_setaffinity(0, {_cpus[(_start + i) % len(_cpus)]
                                 for i in range(min(_NT, len(_cpus)))})
    except (AttributeError, OSError):
        pass

import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
import utils
from omnifold import Multifold, LoadJson
import tensorflow.keras.backend as K

utils.SetStyle()

if _NT > 0:
    tf.config.threading.set_intra_op_parallelism_threads(_NT)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, _NT // 2))

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

parser = argparse.ArgumentParser(description='SBND OmniFold training driver')
parser.add_argument('--config',         default='config_omnifold.json',
                    help='OmniFold settings (JSON)')
parser.add_argument('--plot_folder',    default='./plots/',
                    help='Folder for plots')
parser.add_argument('--weights_folder', default='./weights/',
                    help='Folder to store output weight files')
parser.add_argument('--file_path',      default='FormattedData/',
                    help='Folder containing formatted input .npy files')
parser.add_argument('--nevts',          type=float, default=-1,
                    help='Number of events to use (-1 = all)')
parser.add_argument('--verbose',        action='store_true', default=False,
                    help='Verbose output during training')
parser.add_argument('--shape_only',     action='store_true', default=False,
                    help='Normalise data/MC reco distributions before unfolding')
parser.add_argument('--no_eff',         action='store_true', default=False,
                    help='All truth events are reconstructed (our sample is selected signal only)')
flags = parser.parse_args()

nevts = int(flags.nevts)
opt   = LoadJson(flags.config)

print("\nrun_sbnd.py settings")
print(f"  Config:          {flags.config}")
print(f"  Data path:       {flags.file_path}")
print(f"  Weights folder:  {flags.weights_folder}")
print(f"  no_eff:          {flags.no_eff}")
print(f"  NITER:           {opt.get('NITER', '?')}")
print(f"  NTRIAL:          {opt.get('NTRIAL', '?')} networks averaged per iteration")
print(f"  EPOCHS:          {opt.get('EPOCHS', '?')}")
print(f"  BATCH_SIZE:      {opt.get('BATCH_SIZE', '?')}")
print(f"  NAME:            {opt.get('NAME', '?')}")
print()

if not os.path.exists(flags.plot_folder):
    os.makedirs(flags.plot_folder)

data, mc_reco, mc_gen, reco_mask, gen_mask, \
    data_weights, mc_weights, mc_weights_reco = \
    utils.DataLoader(flags.file_path, opt, nevts)

if flags.shape_only:
    mc_weights_reco *= np.sum(data_weights) / np.sum(mc_weights_reco)

if flags.no_eff:
    mc_gen         = mc_gen[gen_mask]
    mc_weights     = mc_weights[gen_mask]
    gen_mask       = gen_mask[gen_mask]

K.clear_session()
mfold = Multifold(
    version='{}'.format(opt['NAME']),
    verbose=flags.verbose,
    config_file=flags.config,
    plot_folder=flags.plot_folder,
    weights_folder=flags.weights_folder,
)
mfold.mc_gen  = mc_gen
mfold.mc_reco = mc_reco
mfold.data    = data
mfold.Preprocessing(
    weights_mc_reco=mc_weights_reco,
    weights_mc=mc_weights,
    weights_data=data_weights,
    pass_reco=reco_mask,
    pass_gen=gen_mask,
)
mfold.Unfold()