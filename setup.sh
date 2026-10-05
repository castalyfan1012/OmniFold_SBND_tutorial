#!/bin/bash
# Python environment for OmniFold (TensorFlow 2.15, CPU).
#
#   source setup.sh             activate the environment
#   source setup.sh --install   build venv_omnifold/ inside this repository (once, ~10 min)
#
# Which environment is activated:
#   1. $OMNIFOLD_VENV, if set
#   2. venv_omnifold/ inside this repository, if it exists
#   3. otherwise the shared one built for the tutorial

SHARED_VENV=/exp/sbnd/app/users/castalyf/OmniFold_SBND_tutorial/venv_omnifold
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$1" == "--install" ]]; then
    VENV_DIR="${REPO_DIR}/venv_omnifold"
elif [[ -n "$OMNIFOLD_VENV" ]]; then
    VENV_DIR="$OMNIFOLD_VENV"
elif [[ -d "${REPO_DIR}/venv_omnifold" ]]; then
    VENV_DIR="${REPO_DIR}/venv_omnifold"
else
    VENV_DIR="$SHARED_VENV"
fi
export OMNIFOLD_VENV="${VENV_DIR}"
export VENV_PYTHON="${VENV_DIR}/bin/python3"
export TF_CPP_MIN_LOG_LEVEL=1   # hide TensorFlow info messages

if [[ "$1" == "--install" ]]; then
    PY3=""
    for candidate in /usr/bin/python3 /usr/local/bin/python3 $(command -v python3 2>/dev/null); do
        if [[ -x "$candidate" ]]; then PY3="$candidate"; break; fi
    done
    if [[ -z "$PY3" ]]; then echo "ERROR: python3 not found."; return 1; fi
    echo "Building ${VENV_DIR} with $PY3 ($($PY3 --version 2>&1))"

    [[ -d "$VENV_DIR" ]] && rm -rf "$VENV_DIR"
    $PY3 -m venv "$VENV_DIR"
    "${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel
    "${VENV_DIR}/bin/pip" install numpy scipy matplotlib scikit-learn pandas \
        tensorflow-cpu==2.15.0 pyyaml tqdm h5py tables
    if grep -q "Scientific Linux" /etc/redhat-release 2>/dev/null; then
        "${VENV_DIR}/bin/pip" install "urllib3<2"
    fi

    TF_VER=$("${VENV_PYTHON}" -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null)
    if [[ "$TF_VER" != 2* ]]; then echo "ERROR: TensorFlow import failed: ${TF_VER}"; return 1; fi
    echo "Done (TensorFlow ${TF_VER}). Now run: source setup.sh"
    return 0
fi

if [[ ! -f "${VENV_PYTHON}" ]]; then
    echo "ERROR: no environment at ${VENV_DIR}. Run: source setup.sh --install"
    return 1
fi

source "${VENV_DIR}/bin/activate"
export PATH="${VENV_DIR}/bin:${PATH}"   # keep the venv ahead of conda on EAF
hash -r
echo "python3: $(which python3)   TensorFlow: $(python3 -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null)"
