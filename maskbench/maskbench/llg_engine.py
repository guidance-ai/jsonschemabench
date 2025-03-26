from .engine import Engine

import llguidance as llg
from llguidance.numpy import fill_next_token_bitmask, allocate_token_bitmask
import llguidance.hf
import json


class LlgEngine(Engine):

    def __init__(self):
        super().__init__()

    def init(self):
        self.llg_tokenizer = llguidance.hf.from_tokenizer(self.tokenizer)
        self.mask_data = allocate_token_bitmask(
            1, self.llg_tokenizer.vocab_size
        )

    def get_id(self):
        return "llg"

    def get_name(self):
        return "LLGuidance"

    def get_module(self):
        return "llguidance"

    def compile_grammar(self, schema: dict):
        grammars = json.dumps({"grammars": [{"json_schema": schema}]})
        self.interp0 = llg.LLMatcher(self.llg_tokenizer, grammars)
        self.interp = self.interp0
        if self.interp.is_error():
            raise ValueError(self.interp.get_error())

    def reset(self):
        self.interp = self.interp0.deep_copy()

    def compute_mask(self):
        fill_next_token_bitmask(self.interp, self.mask_data, 0)

    def commit_token(self, t: int) -> bool:
        ok = (self.mask_data[0, t // 32] & (1 << (t % 32))) != 0
        if ok:
            self.interp.consume_token(t)
        elif self.interp.is_error():
            raise ValueError(self.interp.get_error())
        return ok
