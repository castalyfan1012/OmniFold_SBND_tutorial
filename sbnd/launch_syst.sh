#!/bin/bash
# Train the systematic universes chosen by `RunStudies.py screen`, split over
# N parallel workers. Safe to re-run: finished universes are skipped and
# crashed ones are retried.
#
#   bash sbnd/launch_syst.sh [N_WORKERS] [THREADS_PER_WORKER] [--with-ml]
#   bash sbnd/launch_syst.sh 3 2 --with-ml
#
# Keep N_WORKERS x THREADS_PER_WORKER <= number of CPUs (nproc).
# --with-ml also trains the 50 ML replicas on the tilt_p_alpha0.3 fake data.

N=${1:-3}
T=${2:-2}
WITH_ML=0
[[ "$3" == "--with-ml" ]] && WITH_ML=1

if pgrep -f "RunStudies.py run-syst" > /dev/null; then
    echo "Workers are already running. Stop them first:"
    echo "  pkill -f 'RunStudies.py run-syst'; pkill -f run_sbnd.py"
    exit 1
fi

mkdir -p logs
echo "CPUs: $(nproc)   process limit: $(ulimit -u)   starting ${N} workers x ${T} threads"
for i in $(seq 0 $((N-1))); do
    nohup python3 sbnd/RunStudies.py run-syst --source screened \
        --worker "$i" --n-workers "$N" --threads "$T" > "logs/syst_w${i}.log" 2>&1 &
    sleep 2
done
if [[ $WITH_ML == 1 ]]; then
    nohup python3 sbnd/RunStudies.py run-ml-unc --n-replicas 50 --var true_p --alpha 0.3 \
        --threads "$T" > logs/ml_tilt_p_alpha0.3.log 2>&1 &
fi

sleep 30
N_UP=$(pgrep -fc "RunStudies.py run-syst")
if [[ "$N_UP" -eq 0 ]]; then
    echo "No worker is running 30 s after launch. Check: tail logs/syst_w0.log; df -h \$HOME"
    exit 1
fi
echo "${N_UP} worker(s) running. Progress: python3 sbnd/RunStudies.py status"
