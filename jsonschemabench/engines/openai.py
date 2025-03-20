import os
from time import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from core.registry import register_engine
from core.engine import Engine, EngineConfig
from core.evaluator import is_json_schema_valid
from core.types import (
    Token,
    CompileStatus,
    DecodingStatus,
    GenerationOutput,
    CompileStatusCode,
    DecodingStatusCode,
)


@dataclass
class OpenAIConfig(EngineConfig):
    model: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class OpenAIEngine(Engine[OpenAIConfig]):
    name = "openai"

    def __init__(
        self,
        config: OpenAIConfig,
        base_url: Optional[str] = None,
        api_key_variable_name: Optional[str] = "OPENAI_API_KEY",
    ):
        super().__init__(config)

        from openai import OpenAI
        from tiktoken import encoding_for_model

        self.client = OpenAI(
            api_key=os.getenv(api_key_variable_name),
            base_url=base_url,
        )
        self.tokenizer = (
            encoding_for_model(self.config.model) if base_url is None else None
        )

    def _generate(self, output: GenerationOutput) -> None:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=output.messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"schema": output.schema, "name": "json_schema"},
                },
                stream=True,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream_options={"include_usage": True},
            )
        except Exception as e:
            output.metadata.compile_status = CompileStatus(
                code=CompileStatusCode.UNSUPPORTED_SCHEMA, message=str(e)
            )
            return

        tokens_str: List[str] = []
        for i, chunk in enumerate(response):
            if i == 0:
                first_token_arrival_time = time()

            if len(chunk.choices) == 0 or chunk.choices[0].finish_reason is not None:
                continue

            chunk_content = chunk.choices[0].delta.content
            if chunk_content == "":
                continue

            tokens_str.append(chunk_content)

        output.token_usage.output_tokens = chunk.usage.completion_tokens
        output.metadata.first_token_arrival_time = first_token_arrival_time
        output.metadata.compile_status = CompileStatus(code=CompileStatusCode.OK)
        output.metadata.decoding_status = DecodingStatus(code=DecodingStatusCode.OK)

        output.generation = "".join(tokens_str)
        output.generated_tokens = [
            Token(id=self.convert_token_to_id(token), text=token)
            for token in tokens_str
        ]
        return

    def adapt_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        recursively_set_additional_properties_false(schema)
        add_root_type_if_missing(schema)
        schema = set_all_properties_required(schema)
        if not is_json_schema_valid(schema):
            print("The JSON schema after adaptation is no longer valid.")
        return schema

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    @property
    def max_context_length(self):
        max_context_length_dict = {
            "gpt-4o": 128 * 1000,
            "gpt-4o-mini": 128 * 1000,
        }
        return max_context_length_dict[self.config.model]


def add_root_type_if_missing(schema: dict):
    if "type" not in schema:
        schema["type"] = "object"


def recursively_set_additional_properties_false(schema: dict):
    if not isinstance(schema, dict):
        return
    if (
        "additionalProperties" not in schema or schema["additionalProperties"]
    ) and schema.get("properties"):
        schema["additionalProperties"] = False
    if "properties" in schema:
        for prop in schema["properties"]:
            recursively_set_additional_properties_false(schema["properties"][prop])
    if "items" in schema:
        recursively_set_additional_properties_false(schema["items"])


def set_all_properties_required(schema: object) -> object:
    if not isinstance(schema, dict):
        return schema
    if "properties" in schema:
        schema["required"] = list(schema["properties"].keys())
    for value in schema.values():
        if isinstance(value, dict):
            set_all_properties_required(value)
        elif isinstance(value, list):
            for item in value:
                set_all_properties_required(item)
    return schema


register_engine(OpenAIEngine, OpenAIConfig)
