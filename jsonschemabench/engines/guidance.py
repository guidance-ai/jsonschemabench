import stopit
from time import time
from typing import List, Optional
from dataclasses import dataclass

from core.registry import register_engine
from engines.llama_cpp import LlamaCppConfig
from core.engine import Engine, EngineConfig
from engines.llama_cpp import LlamaCppEngine
from core.evaluator import is_json_schema_valid
from core.utils import COMPILATION_TIMEOUT, GENERATION_TIMEOUT, safe_min
from core.types import (
    Schema,
    CompileStatus,
    DecodingStatus,
    GenerationOutput,
    CompileStatusCode,
    DecodingStatusCode,
)


@dataclass
class GuidanceConfig(EngineConfig):
    model_engine_config: LlamaCppConfig
    max_tokens: Optional[int] = None
    whitespace_flexible: bool = False


class GuidanceEngine(Engine[GuidanceConfig]):
    name = "guidance"

    def __init__(self, config: GuidanceConfig):
        super().__init__(config)

        from llama_cpp import Llama
        from guidance.models import LlamaCpp

        self.model = Llama.from_pretrained(
            self.config.model_engine_config.model,
            filename=self.config.model_engine_config.filename,
            n_ctx=self.config.model_engine_config.n_ctx,
            verbose=self.config.model_engine_config.verbose,
            n_gpu_layers=self.config.model_engine_config.n_gpu_layers,
        )

        self.guidance_model_state = LlamaCpp(self.model, echo=False)
        
        self.tokenizer = self.guidance_model_state.engine.tokenizer
        self.formatter = LlamaCppEngine.get_chat_formatter(self.model)

    def _generate(self, output: GenerationOutput) -> None:
        from guidance import json as guidance_json

        input = self.formatter(messages=output.messages)
        output.token_usage.input_tokens = self.count_tokens(input)

        try:
            with stopit.ThreadingTimeout(COMPILATION_TIMEOUT) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    generation_op = guidance_json(
                        schema=output.schema,
                        name="generated_object",
                        temperature=self.config.model_engine_config.temperature,
                        max_tokens=safe_min(
                            self.config.model_engine_config.n_ctx
                            - self.count_tokens(input),
                            self.config.max_tokens,
                        ),
                        whitespace_flexible=self.config.whitespace_flexible,
                    )
                    output.metadata.grammar_compilation_end_time = time()
                    output.metadata.compile_status = CompileStatus(
                        code=CompileStatusCode.OK
                    )

            if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                output.metadata.compile_status = CompileStatus(
                    code=CompileStatusCode.COMPILE_TIMEOUT,
                    message="Schema compilation timed out",
                )
                return

        except Exception as e:
            output.metadata.compile_status = CompileStatus(
                code=CompileStatusCode.UNSUPPORTED_SCHEMA, message=str(e)
            )
            return

        try:
            with stopit.ThreadingTimeout(GENERATION_TIMEOUT) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    state_iterator = (
                        self.guidance_model_state.stream() + input + generation_op
                    )
                    for i, guidance_state in enumerate(state_iterator):
                        if i == 0:
                            output.metadata.first_token_arrival_time = time()

            if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                output.metadata.decoding_status = DecodingStatus(
                    code=DecodingStatusCode.DECODING_TIMEOUT,
                    message="Generation timed out",
                )

                # unset the first token arrival time avoid false performance metrics
                output.metadata.first_token_arrival_time = None
                return

        except Exception as e:
            output.metadata.decoding_status = DecodingStatus(
                code=DecodingStatusCode.UNKOWN_ERROR, message=str(e)
            )
            return

        try:
            generation = guidance_state["generated_object"]
            output.metadata.decoding_status = DecodingStatus(code=DecodingStatusCode.OK)
        except KeyError:
            output.metadata.decoding_status = DecodingStatus(
                code=DecodingStatusCode.UNKOWN_ERROR,
                message="Failed to extract generated object",
            )
            generation = ""

        output.generation = generation
        output.token_usage.output_tokens = self.count_tokens(generation)

        return

    def encode(self, text: str) -> Optional[List[int]]:
        return self.tokenizer.encode(text.encode("utf-8"))

    def decode(self, ids: List[int]) -> Optional[str]:
        return self.tokenizer.decode(ids).decode("utf-8")

    def adapt_schema(self, schema: Schema) -> Schema:
        if "type" not in schema:
            schema["type"] = "object"

        if not is_json_schema_valid(schema):
            print("The JSON schema after adaptation is no longer valid.")
        return schema

    @property
    def max_context_length(self) -> int:
        return self.model.n_ctx()

    def close(self):
        self.model.close()


register_engine(GuidanceEngine, GuidanceConfig)
