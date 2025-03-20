from json import loads
from dataclasses import dataclass
from datasets import load_dataset
from typing import Callable, Iterator, Tuple, Optional, List

from core.types import Schema
from core.messages import Message, MessagesFormatter

DATASET_SCHEMA_COLUMN = "json_schema"
DATASET_HUGGINGFACE_PATH = "epfl-dlab/JSONSchemaBench"

DATASET_NAMES = [
    "Github_easy",
    "Github_hard",
    "Github_medium",
    "Github_trivial",
    "Github_ultra",
    "Glaiveai2K",
    "JsonSchemaStore",
    "Kubernetes",
    "Snowplow",
    "WashingtonPost",
    "default",
]


@dataclass
class DatasetConfig:
    dataset_name: str
    limit: Optional[int] = None


class Dataset:
    def __init__(self, config: DatasetConfig):
        """Represents the dataset that is used to benchmark the engine.

        :param config: DatasetConfig
            The configuration for the dataset.
        """
        self.config = config
        self.dataset = load_dataset(
            path=DATASET_HUGGINGFACE_PATH, name=config.dataset_name, split="test"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Schema:
        return loads(self.dataset[idx][DATASET_SCHEMA_COLUMN])

    def filter(self, filter_fn: Callable[[Schema], bool]) -> None:
        self.dataset = self.dataset.filter(
            lambda x: filter_fn(loads(x[DATASET_SCHEMA_COLUMN]))
        )

    def map(self, map_fn: Callable[[Schema], Schema]) -> None:
        self.dataset = self.dataset.map(
            lambda x: map_fn(loads(x[DATASET_SCHEMA_COLUMN]))
        )

    def shuffle(self) -> None:
        self.dataset = self.dataset.shuffle()

    def iter(
        self, messages_formatter: MessagesFormatter
    ) -> Iterator[Tuple[List[Message], Schema]]:
        iterator = (
            self.dataset
            if self.config.limit is None
            else self.dataset.take(self.config.limit)
        )
        for item in iterator:
            schema = loads(item[DATASET_SCHEMA_COLUMN])
            yield messages_formatter(self.config.dataset_name, schema), schema
