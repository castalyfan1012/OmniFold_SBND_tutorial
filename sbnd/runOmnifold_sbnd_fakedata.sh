#!/bin/bash
# Fake-data test: unfold MC that was reweighted with a known tilt, and check that
# OmniFold recovers the tilted truth. Make the fake data first with
#   python3 sbnd/RunStudies.py make-fakedata --mode tilt --var <var> --alpha <alpha>
#
#   nohup bash sbnd/runOmnifold_sbnd_fakedata.sh --var true_p --alpha 0.3 &
#   nohup bash sbnd/runOmnifold_sbnd_fakedata.sh --var both   --alpha 0.3 &
#   nohup bash sbnd/runOmnifold_sbnd_fakedata.sh --var true_p --alpha 0.3 --niter 5 --ntrial 1 &   # quick
#
#   --var     which truth variable was tilted: true_p, true_ke, true_costheta, both
#   --alpha   tilt strength (must match make-fakedata)
#   --niter   OmniFold iterations (default 10)
#   --ntrial  networks averaged per iteration (default 3)
#
# The tag is tilt_<p|ke|costheta|both>_alpha<alpha>, e.g. tilt_p_alpha0.3.
# Output: $SBND_RUNS_DIR/weights_sbnd_fakedata_<tag>/, log: logs/fdt_<tag>.log

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${OMNIFOLD_VENV:-${REPO_DIR}/venv_omnifold}/bin/python3"
[[ -x "$PY" ]] || { echo "No Python environment found. Run: source setup.sh"; exit 1; }

VAR="true_p"
ALPHA="0.3"
NITER=10
NTRIAL=3
while [[ $# -gt 0 ]]; do
    case "$1" in
        --var)    VAR="$2";    shift 2 ;;
        --alpha)  ALPHA="$2";  shift 2 ;;
        --niter)  NITER="$2";  shift 2 ;;
        --ntrial) NTRIAL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

case "${VAR}" in
    true_p)        TAG="tilt_p_alpha${ALPHA}" ;;
    true_ke)       TAG="tilt_ke_alpha${ALPHA}" ;;
    true_costheta) TAG="tilt_costheta_alpha${ALPHA}" ;;
    both)          TAG="tilt_both_alpha${ALPHA}" ;;
    *) echo "--var must be true_p, true_ke, true_costheta or both"; exit 1 ;;
esac

cd "$REPO_DIR"
DATA_DIR="${SBND_DATA_DIR:-${REPO_DIR}/FormattedData/}"
RUNS="${SBND_RUNS_DIR:-${REPO_DIR}/sbnd/runs}"
OUT_DIR="${RUNS}/weights_sbnd_fakedata_${TAG}"
CONFIG="${RUNS}/configs/config_fakedata_${TAG}.json"
mkdir -p "${OUT_DIR}" "${RUNS}/configs" logs

if [[ ! -f "${DATA_DIR%/}/data_weights_sbnd_fakedata_${TAG}.npy" ]]; then
    echo "No fake data for ${TAG}. Run first:"
    echo "  python3 sbnd/RunStudies.py make-fakedata --mode tilt --var ${VAR} --alpha ${ALPHA}"
    exit 1
fi

cat > "${CONFIG}" << EOF
{
  "FILE_MC_RECO": "mc_vals_reco.npy", "FILE_MC_GEN": "mc_vals_truth.npy",
  "FILE_MC_FLAG_RECO": "mc_pass_reco.npy", "FILE_MC_FLAG_GEN": "mc_pass_truth.npy",
  "FILE_DATA_RECO": "mc_vals_reco.npy", "FILE_DATA_FLAG_RECO": "mc_pass_reco.npy",
  "FILE_DATA_WEIGHT": "data_weights_sbnd_fakedata_${TAG}.npy",
  "FILE_MC_RECO_WEIGHT": "mc_weights_reco.npy", "FILE_MC_GEN_WEIGHT": "mc_weights_truth.npy",
  "NITER": ${NITER}, "NTRIAL": ${NTRIAL}, "LR": 1e-3, "BATCH_SIZE": 512,
  "EPOCHS": 100, "NAME": "sbnd_fakedata_${TAG}", "NPATIENCE": 10
}
EOF

echo "Fake-data training ${TAG} (NITER=${NITER}, NTRIAL=${NTRIAL}) -> ${OUT_DIR}/   log: logs/fdt_${TAG}.log"
exec >> "logs/fdt_${TAG}.log" 2>&1
"$PY" run_sbnd.py --config "${CONFIG}" --file_path "${DATA_DIR}" \
    --weights_folder "${OUT_DIR}/" --plot_folder "${OUT_DIR}/" --no_eff --verbose
status=$?
if [[ $status -eq 0 ]]; then echo "Done. Next: python3 sbnd/MakePlots.py validation --var ${VAR} --alpha ${ALPHA}"; else echo "Training failed (exit code $status)"; fi
exit $status
