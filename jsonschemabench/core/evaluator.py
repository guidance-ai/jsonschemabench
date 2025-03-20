from uuid import UUID
from json import loads
from prettytable import PrettyTable
from typing import List, Tuple, Optional
from ipaddress import IPv4Address, IPv6Address
from jsonschema import Draft202012Validator, FormatChecker, SchemaError

from core.utils import safe_divide, median, detect_none
from core.types import Schema, CompileStatusCode, GenerationOutput, PerfMetrics


def is_json_schema_valid(schema: Schema):
    try:
        Draft202012Validator.check_schema(schema)
        return True
    except SchemaError:
        return False


format_checker = FormatChecker()


@format_checker.checks("ipv4")
def ipv4_check(value):
    IPv4Address(value)


@format_checker.checks("ipv6")
def ipv6_check(value):
    IPv6Address(value)


@format_checker.checks("uuid")
def uuid_check(value):
    UUID(value)


def validate_json_schema(instance: Schema, schema: Schema) -> bool:
    if not is_json_schema_valid(schema):
        return False
    validator = Draft202012Validator(schema, format_checker=format_checker)
    try:
        validator.validate(instance)

    # we catch all exceptions include ValidationError and Error from extension validators
    except Exception:
        return False
    return True


def evaluate(
    outputs: List[GenerationOutput],
) -> Tuple[Optional[float], Optional[float], Optional[float], PerfMetrics]:
    declared_coverage = 0
    empirical_coverage = 0

    for generation_output in outputs:
        generation = generation_output.generation
        schema = generation_output.schema

        if schema is None or generation is None:
            continue

        if generation_output.metadata.compile_status.code == CompileStatusCode.OK:
            declared_coverage += 1

        try:
            json_object = loads(generation)
        except Exception:
            continue

        if not validate_json_schema(json_object, schema):
            continue

        empirical_coverage += 1

    ttft_list = [
        generation_output.perf_metrics.ttft
        for generation_output in outputs
        if generation_output.perf_metrics.ttft is not None
    ]
    tpot_list = [
        generation_output.perf_metrics.tpot
        for generation_output in outputs
        if generation_output.perf_metrics.tpot is not None
    ]
    tgt_list = [
        generation_output.perf_metrics.tgt
        for generation_output in outputs
        if generation_output.perf_metrics.tgt is not None
    ]
    gct_list = [
        generation_output.perf_metrics.gct
        for generation_output in outputs
        if generation_output.perf_metrics.gct is not None
    ]

    return (
        safe_divide(declared_coverage, len(outputs)),
        safe_divide(empirical_coverage, len(outputs)),
        safe_divide(empirical_coverage, declared_coverage),
        PerfMetrics(
            ttft=median(ttft_list),
            tpot=median(tpot_list),
            tgt=median(tgt_list),
            gct=median(gct_list),
        ),
    )


def print_scores(
    declared_coverage: List[Optional[float]],
    empirical_coverage: List[Optional[float]],
    compliance: List[Optional[float]],
    perf_metrics: List[PerfMetrics],
    tasks: List[str],
) -> None:
    table = PrettyTable(
        [
            "Task",
            "Declared coverage",
            "Empirical coverage",
            "Compliance",
            "TTFT (s)",
            "TPOT (ms)",
            "TGT (s)",
            "GCT (s)",
        ]
    )
    for task, dc, ec, cl, pm in zip(
        tasks, declared_coverage, empirical_coverage, compliance, perf_metrics
    ):
        table.add_row(
            [
                task,
                detect_none(dc),
                detect_none(ec),
                detect_none(cl),
                detect_none(pm.ttft),
                detect_none(pm.tpot),
                detect_none(pm.tgt),
                detect_none(pm.gct),
            ]
        )
    print(table)
