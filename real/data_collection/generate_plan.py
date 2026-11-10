#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the paired-collection plan CSV (3 tasks per collection group).

Tasks are named taskX-Y: X is the scene number, Y the task number within
that scene (1..3). Every group has 6 rows (3 tasks x 2 recordings); within a
task the robot-arm row comes first and its human row follows immediately:

    robot_taskX-1, human_taskX-1, robot_taskX-2, human_taskX-2, robot_taskX-3, human_taskX-3

The naming drives pairing in convert/split_dataset.py:
  - a task starting with robot_ -> the robot (arm) dataset;
  - a task starting with human_ -> the human (hand) dataset;
  - a robot row pairs with the human row immediately after it once the prefix
    (robot_/human_) is stripped from both labels.
So when changing the naming, keep robot_ + task for the robot row and
human_ + task for the human row, and make sure the task part itself does not
start with robot_ or human_.

scene column: groups alternate scene1, scene1_alt, scene2, scene2_alt, ...
-- the two groups of a scene are its original placement and a moved placement;
the label is written only on the group's first row. Operators set up the table
according to this column; during processing it drives conditions_list.json: the
episodes sharing the same base scene (trailing _alt stripped), the same
task and a plan-row distance <= 30 share one set of reference conditions, so
the same task under scene1 and scene1_alt are each other's condition
(exactly 2, i.e. --max-conditions 2 in convert.sh).

data_path column: after recording each episode the collection program writes
that episode's parquet path back into this column (see write_plan_data in
lerobot/scripts/control_robot.py); processing uses it to match episodes to
plan rows.

Usage:
    python generate_plan.py -o plan.csv -g 4      # 4 groups = 2 scenes x 2 placements
"""

import argparse
import csv

HEADER = ["index", "scene", "objects", "action", "data_path"]

# Performer prefixes; keep the `robot_/human_ + task` naming rule when changing them.
ROBOT_PREFIX = "robot_"
HUMAN_PREFIX = "human_"

# Tasks per scene/placement; names are `taskX-1` .. `taskX-{N}`.
TASKS_PER_SCENE = 3

# Scene-name suffix of a scene's second placement (see the module docstring).
SCENE_MOVE_SUFFIX = "_alt"


def scene_label(group: int) -> str:
    """Scene label of the 0-based `group`: scene1, scene1_alt, scene2, ..."""
    scene_id = group // 2 + 1
    label = f"scene{scene_id}"
    if group % 2 == 1:
        label += SCENE_MOVE_SUFFIX
    return label


def group_actions(scene_id: int) -> list[str]:
    """Action sequence of one group: robot then human per task, TASKS_PER_SCENE tasks."""
    actions = []
    for k in range(1, TASKS_PER_SCENE + 1):
        task = f"task{scene_id}-{k}"
        actions.append(f"{ROBOT_PREFIX}{task}")
        actions.append(f"{HUMAN_PREFIX}{task}")
    return actions


def build_rows(n_groups: int) -> list[list[str]]:
    rows = []
    seq = 1
    for group in range(n_groups):
        scene = scene_label(group)
        scene_id = group // 2 + 1
        for i, action in enumerate(group_actions(scene_id)):
            scene_cell = scene if i == 0 else ""
            rows.append([str(seq), scene_cell, "", action, ""])
            seq += 1
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate the collection plan CSV (3 tasks per group, one robot + one human row each)"
    )
    parser.add_argument("-o", "--output", default="plan.csv",
                        help="output CSV path")
    parser.add_argument("-g", "--groups", type=int, default=50,
                        help=f"number of collection groups (default 50; {2 * TASKS_PER_SCENE} rows per group, "
                             "every 2 groups are the two placements of one scene)")
    parser.add_argument("--no-header", action="store_true", help="do not write the header row")
    args = parser.parse_args()

    rows = build_rows(args.groups)

    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if not args.no_header:
            writer.writerow(HEADER)
        writer.writerows(rows)

    n_scenes = (args.groups + 1) // 2
    print(f"Generated {len(rows)} plan rows ({args.groups} groups x {TASKS_PER_SCENE} tasks x 2 entries, "
          f"{n_scenes} scenes x 2 placements) -> {args.output}")


if __name__ == "__main__":
    main()
