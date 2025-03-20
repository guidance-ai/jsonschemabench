from json import loads
from typing import Dict, List
from argparse import ArgumentParser
from dacite import from_dict, Config

from core.types import GenerationOutput
from core.evaluator import evaluate, print_scores


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--outputs", type=str, required=True)
    args = parser.parse_args()

    dacite_config = Config(check_types=False)
    with open(args.outputs, "r") as f:
        engine_config = loads(f.readline())
        outputs = [
            from_dict(GenerationOutput, loads(line), config=dacite_config)
            for line in f[1:]
        ]

    task_outputs: Dict[str, List[GenerationOutput]] = {}
    for output in outputs:
        if output.task not in task_outputs:
            task_outputs[output.task] = []
        task_outputs[output.task].append(output)

    compliance = []
    perf_metrics = []
    declared_coverage = []
    empirical_coverage = []
    for outputs in task_outputs.values():
        dc, ec, cl, pm = evaluate(outputs)

        compliance.append(cl)
        perf_metrics.append(pm)
        declared_coverage.append(dc)
        empirical_coverage.append(ec)

    print(engine_config)
    print_scores(
        declared_coverage,
        empirical_coverage,
        compliance,
        perf_metrics,
        list(task_outputs.keys()),
    )
