#!/bin/bash
# Closure test: unfold the nominal MC onto itself. The result should give back
# the nominal truth spectrum (all weights ~1).
#
#   nohup bash sbnd/runOmnifold_sbnd_closure.sh &                       # 10 iterations, 3 networks
#   nohup bash sbnd/runOmnifold_sbnd_closure.sh --ntrial 7 &            # more networks, less noise, ~2x slower
#
# Output: $SBND_RUNS_DIR/weights_sbnd_closure/ (default sbnd/runs), log: logs/closure.log

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${OMNIFOLD_VENV:-${REPO_DIR}/venv_omnifold}/bin/python3"
[[ -x "$PY" ]] || { echo "No Python environment found. Run: source setup.sh"; exit 1; }

NITER=10
NTRIAL=3
while [[ $# -gt 0 ]]; do
    case "$1" in
        --niter)  NITER="$2";  shift 2 ;;
        --ntrial) NTRIAL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

cd "$REPO_DIR"
DATA_DIR="${SBND_DATA_DIR:-${REPO_DIR}/FormattedData/}"
RUNS="${SBND_RUNS_DIR:-${REPO_DIR}/sbnd/runs}"
OUT_DIR="${RUNS}/weights_sbnd_closure"
CONFIG="${RUNS}/configs/config_closure.json"
mkdir -p "${OUT_DIR}" "${RUNS}/configs" logs

cat > "${CONFIG}" << EOF
{
  "FILE_MC_RECO": "mc_vals_reco.npy", "FILE_MC_GEN": "mc_vals_truth.npy",
  "FILE_MC_FLAG_RECO": "mc_pass_reco.npy", "FILE_MC_FLAG_GEN": "mc_pass_truth.npy",
  "FILE_DATA_RECO": "mc_vals_reco.npy", "FILE_DATA_FLAG_RECO": "mc_pass_reco.npy",
  "FILE_DATA_WEIGHT": "mc_weights_reco.npy",
  "FILE_MC_RECO_WEIGHT": "mc_weights_reco.npy", "FILE_MC_GEN_WEIGHT": "mc_weights_truth.npy",
  "NITER": ${NITER}, "NTRIAL": ${NTRIAL}, "LR": 1e-3, "BATCH_SIZE": 512,
  "EPOCHS": 100, "NAME": "sbnd_nueCC_closure", "NPATIENCE": 10
}
EOF

echo "Closure training (NITER=${NITER}, NTRIAL=${NTRIAL}) -> ${OUT_DIR}/   log: logs/closure.log"
exec >> logs/closure.log 2>&1
"$PY" run_sbnd.py --config "${CONFIG}" --file_path "${DATA_DIR}" \
    --weights_folder "${OUT_DIR}/" --plot_folder "${OUT_DIR}/" --no_eff --verbose
status=$?
if [[ $status -eq 0 ]]; then echo "Done. Next: python3 sbnd/RunStudies.py check-closure"; else echo "Training failed (exit code $status)"; fi
exit $status