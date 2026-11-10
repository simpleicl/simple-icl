#!/usr/bin/env bash
# Replay a recorded episode: feed the recorded actions frame by frame to the
# Franka arm (replay mode) and print error statistics vs the recorded states.
#
# Usage (run from the repo root):
#     DATASET=outputs/1015 EPISODE=0 MOCK=1 bash replay.sh      # mock replay (no hardware)
#     DATASET=outputs/1015 EPISODE=0 bash replay.sh              # real-arm replay
#
# Environment variables:
#     DATASET  Dataset root (image-format dataset with meta/info.json); required, e.g. outputs/1015
#     EPISODE  Episode index to replay; default 0
#     MOCK=1   Use the mock arm + cameras (default: real arm)
#     EXTRA    Extra arguments passed through to replay_franka.py (e.g. EXTRA="--fps 5")
#     PYTHON   Override the python interpreter (default: python)
#
# Replay output is also saved to outputs/logfiles/replay-<DATASET name>-ep<N>.log.
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-}"
EPISODE="${EPISODE:-0}"

if [ -z "${DATASET}" ]; then
    echo "Error: please set DATASET (dataset root, e.g. DATASET=outputs/1015)" >&2
    exit 1
fi

if [ ! -f "${DATASET}/meta/info.json" ]; then
    echo "Error: dataset not found: ${DATASET}/meta/info.json" >&2
    exit 1
fi

MOCK_ARGS=(--no-mock)
if [ -n "${MOCK:-}" ]; then
    MOCK_ARGS=(--mock)
fi

LOG_DIR="outputs/logfiles"
mkdir -p "${LOG_DIR}"
DS_NAME="$(basename "${DATASET}")"
LOG_FILE="${LOG_DIR}/replay-${DS_NAME}-ep${EPISODE}.log"

echo "Replaying ${DATASET} episode ${EPISODE} (log: ${LOG_FILE})"
PYTHONPATH="$(pwd)" "${PYTHON}" lerobot/scripts/replay_franka.py \
    --root "${DATASET}" \
    --episode "${EPISODE}" \
    "${MOCK_ARGS[@]}" \
    ${EXTRA:-} 2>&1 | tee "${LOG_FILE}"
