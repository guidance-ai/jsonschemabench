import os
from core.bench import bench
from argparse import ArgumentParser
from core.dataset import DATASET_NAMES
from core.utils import load_config, disable_print
from core.registry import ENGINE_TO_CLASS, ENGINE_TO_CONFIG


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--engine", type=str, required=True, choices=ENGINE_TO_CLASS.keys()
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--limit", type=int, required=False)
    parser.add_argument("--save_outputs", action="store_true")
    args = parser.parse_args()

    tasks = args.tasks.split(",")
    if not all(task in DATASET_NAMES for task in tasks):
        raise ValueError(
            f"Invalid task names: {args.tasks}, available: {DATASET_NAMES}"
        )

    if args.config is None:
        args.config = os.path.join("tests/configs", f"{args.engine}.yaml")

    with disable_print():
        engine = ENGINE_TO_CLASS[args.engine](
            load_config(ENGINE_TO_CONFIG[args.engine], args.config)
        )

    bench(
        engine=engine,
        tasks=tasks,
        limit=args.limit,
        save_outputs=args.save_outputs,
        close_engine=True,
    )
