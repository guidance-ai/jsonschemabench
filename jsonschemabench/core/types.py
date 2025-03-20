from enum import Enum
from uuid import uuid4
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from core.messages import Message
from core.utils import safe_divide, safe_subtract


Schema = Dict[str, Any]


class CompileStatusCode(int, Enum):
    TBD = -1
    OK = 0
    UNSUPPORTED_SCHEMA = 1
    RUNTIME_GRAMMAR_ERROR = 2
    API_BAD_RESPONSE = 3
    PROMPT_TOO_LONG = 4
    COMPILE_TIMEOUT = 5
    RUNTIME_TIMEOUT = 6
    UNKOWN_ERROR = 7


class DecodingStatusCode(int, Enum):
    TBD = -1
    OK = 0
    EXCEEDING_MAX_CTX = 1
    DECODING_TIMEOUT = 2
    BAD_API_RESPONSE = 3
    UNKOWN_ERROR = 4


@dataclass
class CompileStatus:
    code: CompileStatusCode = CompileStatusCode.TBD
    message: Optional[str] = None


@dataclass
class DecodingStatus:
    code: DecodingStatusCode = DecodingStatusCode.TBD
    message: Optional[str] = None


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    def __str__(self) -> str:
        return (
            f"token usage: {self.input_tokens:,} input, {self.output_tokens:,} output."
        )


@dataclass
class Token:
    id: Optional[int] = None
    text: Optional[str] = None
    logprob: Optional[float] = None


@dataclass
class GenerationMetadata:
    first_token_arrival_time: Optional[float] = None
    grammar_compilation_end_time: Optional[float] = None
    compile_status: Optional[CompileStatus] = field(default_factory=CompileStatus)
    decoding_status: Optional[DecodingStatus] = field(default_factory=DecodingStatus)


@dataclass
class PerfMetrics:
    """Performance metrics for generation processes."""

    # Time to first token in s
    ttft: Optional[float] = None
    # Time per output token in ms
    tpot: Optional[float] = None
    # Total generation time in s
    tgt: Optional[float] = None
    # Grammar compilation time in s
    gct: Optional[float] = None
    # Prefilling time in s
    prft: Optional[float] = None
    # Peak memory in MB
    peak_memory: Optional[float] = None

    @classmethod
    def from_timestamps(
        cls,
        start_time: float,
        grammar_compilation_end_time: Optional[float],
        first_token_arrival_time: Optional[float],
        end_time: float,
        num_output_tokens: int,
    ):
        ttft = safe_subtract(first_token_arrival_time, start_time)
        tpot = (
            safe_divide(
                safe_subtract(end_time, first_token_arrival_time),
                safe_subtract(num_output_tokens, 1),
            )
            if num_output_tokens > 0
            else None
        )
        tgt = safe_subtract(end_time, start_time)
        gct = safe_subtract(grammar_compilation_end_time, start_time)
        prft = safe_subtract(first_token_arrival_time, grammar_compilation_end_time)
        return cls(
            ttft=ttft,
            tpot=tpot * 1000 if tpot is not None else None,
            tgt=tgt,
            gct=gct,
            prft=prft,
        )


@dataclass
class GenerationOutput:
    """Output of a generation run."""

    task: str
    messages: List[Message]
    generation: str
    schema: Schema
    id: str = field(default_factory=lambda: str(uuid4()))
    generated_tokens: List[Token] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    perf_metrics: PerfMetrics = field(default_factory=PerfMetrics)
    metadata: GenerationMetadata = field(default_factory=GenerationMetadata)
