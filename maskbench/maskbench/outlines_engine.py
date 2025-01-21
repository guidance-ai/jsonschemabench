from .engine import Engine
import json
from outlines.models.transformers import TransformerTokenizer
from outlines_core.fsm.outlines_core_rs import build_regex_from_schema
from outlines_core.fsm.guide import Write, Generate
from outlines.fsm.guide import RegexGuide


class OutlinesEngine(Engine):
    def __init__(self):
        super().__init__()

    def init(self):
        self.outlines_tokenizer = TransformerTokenizer(self.tokenizer)

    def compile_grammar(self, schema: dict):
        rx = build_regex_from_schema(json.dumps(schema))
        self.guide = RegexGuide.from_regex(rx, self.outlines_tokenizer)

    def reset(self):
        self.guide_state = self.guide.initial_state

    def compute_mask(self):
        self.inst = self.guide.get_next_instruction(self.guide_state)

    def commit_token(self, t: int) -> bool:
        ok = False
        inst = self.inst
        if isinstance(inst, Write):
            ok = t in inst.tokens
        elif isinstance(inst, Generate) and inst.tokens is not None:
            ok = t in inst.tokens
        if ok:
            self.guide_state = self.guide.get_next_state(self.guide_state, t)
        return ok
