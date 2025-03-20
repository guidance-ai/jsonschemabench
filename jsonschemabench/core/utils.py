import os
import sys
import random
import string
from json import dumps
from dacite import from_dict
from omegaconf import OmegaConf
from contextlib import contextmanager
from typing import List, Optional, TypeVar, Type

GENERATION_TIMEOUT = 60
COMPILATION_TIMEOUT = 10


T = TypeVar("T")


def load_config(config_type: Type[T], config_path: str) -> T:
    return from_dict(data_class=config_type, data=OmegaConf.load(config_path))


def safe_divide(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Safely divides a by b, returning None if either input is None."""
    if a is None or b is None or b == 0:
        return None
    return a / b


def safe_subtract(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Safely subtracts b from a, returning None if either input is None."""
    if a is None or b is None:
        return None
    return a - b


def safe_min(a: int, b: Optional[int]) -> int:
    if b is None:
        return a
    return min(a, b)


def median(values: List[float]) -> float:
    if len(values) == 0:
        return None
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n % 2 == 0:
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        return sorted_values[n // 2]


def detect_none(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


@contextmanager
def disable_print():
    stdout = sys.stdout
    stderr = sys.stderr
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout = stdout
        sys.stderr = stderr


def nanoid(length: int = 4) -> str:
    return "".join(random.choices(string.ascii_letters, k=length))
