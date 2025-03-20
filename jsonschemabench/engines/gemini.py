import os
from typing import Optional, List

from core.types import Schema
from core.registry import register_engine
from core.evaluator import is_json_schema_valid
from engines.openai import OpenAIEngine, OpenAIConfig


class GeminiEngine(OpenAIEngine):
    name = "gemini"

    def __init__(self, config: OpenAIConfig):
        super().__init__(
            config,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key_variable_name="GEMINI_API_KEY",
        )

        from google.generativeai import GenerativeModel, configure

        configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = GenerativeModel(model_name=self.config.model)

    def encode(self, _: str) -> Optional[List[int]]:
        return None

    def decode(self, _: List[int]) -> Optional[str]:
        return None

    def count_tokens(self, text: str) -> int:
        return self.model.count_tokens(text).total_tokens

    def adapt_schema(self, schema: Schema) -> Schema:
        required_fields = schema.get("required", [])
        if "id" not in required_fields:
            schema.pop("id", None)
        if "title" not in required_fields:
            schema.pop("title", None)
        if "$schema" not in required_fields:
            schema.pop("$schema", None)
        if "$id" not in required_fields:
            schema.pop("$id", None)

        if not is_json_schema_valid(schema):
            print("The JSON schema after adaptation is no longer valid.")
        return schema

    @property
    def max_context_length(self) -> int:
        max_context_length_dict = {
            "models/gemini-2.0-flash": 1_048_576,
            "models/gemini-2.0-flash-lite": 1_048_576,
            "models/gemini-1.5-flash": 1_048_576,
            "models/gemini-1.5-flash-8b": 1_048_576,
            "models/gemini-1.5-pro": 2_097_152,
        }
        return max_context_length_dict[self.config.model]


register_engine(GeminiEngine, OpenAIConfig)
