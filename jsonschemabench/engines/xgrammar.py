import os
import torch
import stopit
from time import time
from json import dumps
from dataclasses import dataclass
from transformers.generation import LogitsProcessor
from typing import List, Optional, TYPE_CHECKING, Dict, Any

from core.registry import register_engine
from core.engine import Engine, EngineConfig
from core.utils import COMPILATION_TIMEOUT, GENERATION_TIMEOUT
from core.types import (
    CompileStatus,
    DecodingStatus,
    GenerationOutput,
    CompileStatusCode,
    DecodingStatusCode,
)


if TYPE_CHECKING:
    from xgrammar import GrammarCompiler


class TimingLogitsProcessor(LogitsProcessor):
    """Logits processor that records timestamps for token generation."""

    def __init__(self):
        super().__init__()
        self.timestamps = []

    def __call__(self, _, scores):
        self.timestamps.append(time())
        return scores


@dataclass
class XGrammarConfig(EngineConfig):
    model: str
    temperature: float = 0
    max_tokens: Optional[int] = 4096
    grammar_cache_enabled: bool = False


class XGrammarEngine(Engine[XGrammarConfig]):
    name = "xgrammar"

    def __init__(self, config: XGrammarConfig):
        super().__init__(config)
        add_environment_variables()

        from xgrammar import TokenizerInfo, GrammarCompiler
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model, torch_dtype=torch.bfloat16
        ).to(get_best_device())
        self.tokenizer.pad_token = self.tokenizer.eos_token

        tokenizer_info = TokenizerInfo.from_huggingface(
            self.tokenizer, vocab_size=self.model.config.vocab_size
        )
        self.grammar_compiler = GrammarCompiler(
            tokenizer_info, cache_enabled=self.config.grammar_cache_enabled
        )

    def _generate(self, output: GenerationOutput) -> None:
        from transformers.generation import GenerationConfig
        from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor

        timing_processor = TimingLogitsProcessor()
        logits_processors = [timing_processor]

        try:
            json_schema_str = dumps(output.schema)

            seg_fault_check = self._check_grammar_safety(
                self.grammar_compiler, json_schema_str, COMPILATION_TIMEOUT
            )
            if not seg_fault_check["success"]:
                if "exit_code" in seg_fault_check and seg_fault_check["exit_code"] == 2:
                    output.metadata.compile_status = CompileStatus(
                        code=CompileStatusCode.UNSUPPORTED_SCHEMA,
                        message="Grammar compilation timed out",
                    )
                    return
                else:
                    output.metadata.compile_status = CompileStatus(
                        code=CompileStatusCode.UNSUPPORTED_SCHEMA,
                        message=seg_fault_check.get(
                            "error",
                            f"Unknown error with exit code {seg_fault_check.get('exit_code', 'unknown')}",
                        ),
                    )
                    return

            with stopit.ThreadingTimeout(COMPILATION_TIMEOUT) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    compiled_grammar = self.grammar_compiler.compile_json_schema(
                        json_schema_str
                    )

                    output.metadata.grammar_compilation_end_time = time()
                    output.metadata.compile_status = CompileStatus(
                        code=CompileStatusCode.OK
                    )
                    logits_processors.append(XGrammarLogitsProcessor(compiled_grammar))

            if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                output.metadata.compile_status = CompileStatus(
                    code=CompileStatusCode.COMPILE_TIMEOUT,
                    message="Grammar compilation timed out",
                )
                return

        except Exception as e:
            output.metadata.compile_status = CompileStatus(
                code=CompileStatusCode.UNSUPPORTED_SCHEMA, message=str(e)
            )
            return

        input = self.tokenizer.apply_chat_template(
            output.messages, tokenize=False, add_generation_prompt=True
        )

        model_input = self.tokenizer(
            input,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
        ).to("cuda")

        input_length = model_input["input_ids"].shape[1]

        try:
            with stopit.ThreadingTimeout(GENERATION_TIMEOUT) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    model_output = self.model.generate(
                        model_input["input_ids"],
                        generation_config=GenerationConfig(
                            max_new_tokens=self.config.max_tokens,
                            temperature=self.config.temperature
                            if self.config.temperature > 0
                            else None,
                            do_sample=self.config.temperature > 0,
                            pad_token_id=self.tokenizer.eos_token_id,
                        ),
                        attention_mask=model_input["attention_mask"],
                        tokenizer=self.tokenizer,
                        logits_processor=logits_processors,
                    )
                    output.metadata.decoding_status = DecodingStatus(
                        code=DecodingStatusCode.OK
                    )

            if len(timing_processor.timestamps) > 0:
                output.metadata.first_token_arrival_time = timing_processor.timestamps[
                    0
                ]

            if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                output.metadata.decoding_status = DecodingStatus(
                    code=DecodingStatusCode.DECODING_TIMEOUT,
                    message="Generation timed out",
                )
                return

        except Exception as e:
            output.metadata.decoding_status = DecodingStatus(
                code=DecodingStatusCode.UNKOWN_ERROR, message=str(e)
            )
            return

        generated_sequences = model_output[:, input_length:]
        generated_texts = self.tokenizer.batch_decode(
            generated_sequences, skip_special_tokens=True
        )

        output_text = generated_texts[0] if generated_texts else ""

        if timing_processor.timestamps:
            output.metadata.first_token_arrival_time = timing_processor.timestamps[0]

        output.generation = output_text
        output.token_usage.output_tokens = self.count_tokens(output_text)

        return

    def _check_grammar_safety(
        self, grammar_compiler: "GrammarCompiler", schema_str: str, timeout: int
    ) -> Dict[str, Any]:
        def child_process():
            import signal

            signal.signal(signal.SIGALRM, lambda _, __: os._exit(2))
            signal.alarm(timeout)
            try:
                grammar_compiler.compile_json_schema(schema_str)
                os._exit(0)
            except Exception:
                os._exit(1)

        id = os.fork()
        if id == 0:
            child_process()
        else:
            _, status = os.waitpid(id, 0)
            if os.WIFEXITED(status):
                exit_code = os.WEXITSTATUS(status)
                return {"success": exit_code == 0, "exit_code": exit_code}
            elif os.WIFSIGNALED(status):
                signal_num = os.WTERMSIG(status)
                return {
                    "success": False,
                    "error": f"Process terminated by signal {signal_num}",
                    "exit_code": -signal_num,
                }
            return {"success": False, "error": "Unknown status"}

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def max_context_length(self) -> int:
        return self.tokenizer.model_max_length


def add_environment_variables():
    import os
    import subprocess

    try:
        result = subprocess.run(
            ["find", "/usr", "-name", "libcuda.so"],
            capture_output=True,
            text=True,
            check=True,
        )

        paths = result.stdout.strip().split("\n")

        if paths and paths[0]:
            libcuda_dir = os.path.dirname(paths[0])
            os.environ["TRITON_LIBCUDA_PATH"] = libcuda_dir
            print(f"Set TRITON_LIBCUDA_PATH to {libcuda_dir}")
        else:
            print("No libcuda.so found in /usr")

    except Exception as e:
        print(f"Error setting TRITON_LIBCUDA_PATH: {e}")

    # Disable tokenizer parallelism to avoid deadlocks
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


register_engine(XGrammarEngine, XGrammarConfig)
