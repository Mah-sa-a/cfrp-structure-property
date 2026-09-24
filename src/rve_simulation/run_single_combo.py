"""
Reads SLURM_ARRAY_TASK_ID, maps it to exactly one (Vf, L_over_D) pair
out of the full grid defined in rve_size_convergence_pipeline.py, and
runs only that one combination via pipeline.run_combo().

--array upper bound MUST equal
    len(pipeline.VF_LIST) * len(pipeline.L_OVER_D_LIST) - 1
Currently: 3 * 6 = 18 combinations -> --array=0-17

Can also be run manually for a single combo without Slurm, e.g. to
test or resume one specific failing task:
    python run_single_combo.py --task_id 17
"""

import argparse
import os
import sys

import rve_size_convergence_pipeline as pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=None,
                         help="Override SLURM_ARRAY_TASK_ID for manual testing.")
    args = parser.parse_args()

    if args.task_id is not None:
        task_id = args.task_id
    else:
        task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id_str is None:
            sys.exit("SLURM_ARRAY_TASK_ID is not set, and --task_id was not "
                      "provided. Nothing to run.")
        task_id = int(task_id_str)

    all_combos = [(Vf, LD) for Vf in pipeline.VF_LIST for LD in pipeline.L_OVER_D_LIST]

    expected_max = len(pipeline.VF_LIST) * len(pipeline.L_OVER_D_LIST) - 1
    if task_id > expected_max:
        sys.exit(f"task_id={task_id} is out of range: only {len(all_combos)} "
                  f"combinations exist (valid range 0-{expected_max}). "
                  f"Check that --array in your Slurm script matches the grid.")

    Vf, L_over_D = all_combos[task_id]
    print(f"Task {task_id}: Vf={Vf}, L_over_D={L_over_D}")

    pipeline.run_combo(Vf, L_over_D, task_id, output_dir=str(pipeline.WORK_DIR))


if __name__ == "__main__":
    main()
