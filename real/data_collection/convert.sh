#!/usr/bin/env bash
# Process the data collected on DATE_C into the paired format (robot arm side +
# human hand side).
#
# Usage (run from the repo root):
#     DATE_C=1015 bash convert.sh
#
# Optional environment variables:
#     MOCK=1             mock demo mode: relax the validity thresholds in
#                        convert/split_dataset.py that are tuned for real data
#                        (see the "mock sample" section of the README).
#     DROP_EPS="3,8-9"   source episode indices to discard (pairs are dropped
#                        whole); may be left unset.
#     PYTHON=...         override the python interpreter (default: python).
#
# Input:  outputs/${DATE_C}/           image-format LeRobot dataset (with plan.csv)
# Output: outputs/videos/${DATE_C}/    video dataset (episode tasks rewritten
#                                      from the plan's action column)
#         outputs/split_new/${DATE_C}/ split result: independent robot/ and
#                                      human/ datasets plus pairing.csv and
#                                      split.log
set -euo pipefail

PYTHON="${PYTHON:-python}"

: "${DATE_C:?Please set DATE_C first}"

SRC_DATA="outputs/${DATE_C}"
DST_DATA="outputs/videos/${DATE_C}"
SPLIT_DATA="outputs/split_new/${DATE_C}"

if [ ! -f "${SRC_DATA}/meta/info.json" ]; then
    echo "Error: dataset not found: ${SRC_DATA}/meta/info.json" >&2
    exit 1
fi

DROP_ARGS=()
if [ -n "${DROP_EPS:-}" ]; then
    DROP_ARGS=(--drop-episodes "${DROP_EPS}")
fi

SPLIT_CHECK_ARGS=()
if [ -n "${MOCK:-}" ]; then
    # Mock data has constant camera frames and a still arm; it cannot pass the
    # duplicate-frame / motion checks tuned for real data.
    SPLIT_CHECK_ARGS=(
        --no-dup-check
        --min-frames-robot 1
        --min-frames-human 1
        --motion-min 0
    )
fi

# 1) Images -> video; rewrite episode tasks from the plan's action column and
#    fill the scene column into plan_filled.csv.
PYTHONPATH="$(pwd)" "${PYTHON}" convert/convert_to_video.py \
    --src_dataset "${SRC_DATA}" --dst_dataset "${DST_DATA}"

# 2) Split and pair by task label (robot_/human_) into the robot/ and human/ datasets.
"${PYTHON}" convert/split_dataset.py \
    --src "${DST_DATA}" --root "${SPLIT_DATA}" \
    ${DROP_ARGS[@]+"${DROP_ARGS[@]}"} \
    --max-conditions 2 \
    ${SPLIT_CHECK_ARGS[@]+"${SPLIT_CHECK_ARGS[@]}"}
