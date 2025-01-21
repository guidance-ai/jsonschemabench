from transformers import PreTrainedTokenizer
from .engine import Engine
from huggingface_hub import hf_hub_download

import llguidance as llg
import json


class LlgEngine(Engine):
    def __init__(self):
        super().__init__()

    def init(self):
        tokenizer_json_path = hf_hub_download(
            repo_id=self.tokenizer_model_id, filename="tokenizer.json"
        )
        with open(tokenizer_json_path, "r") as f:
            self.llg_tokenizer = llg.LLTokenizer(f.read())
        self.mask_data = bytearray(bytearray((self.llg_tokenizer.vocab_size + 7) // 8))

    def get_id(self):
        return "llg"

    def get_name(self):
        return "LLGuidance"

    def get_module(self):
        return "llguidance"

    def compile_grammar(self, schema: dict):
        grammars = json.dumps({"grammars": [{"json_schema": schema}]})
        self.interp0 = llg.LLInterpreter(
            self.llg_tokenizer, grammars, enable_backtrack=False, enable_ff_tokens=False
        )
        self.interp0.start_without_prompt()

    def reset(self):
        self.interp = self.interp0.deep_copy()

    def compute_mask(self):
        res = self.interp.compute_mask_into(self.mask_data)
        # ignore res?

    def commit_token(self, t: int) -> bool:
        ok = (self.mask_data[t // 8] & (1 << (t % 8))) != 0
        if ok:
            self.interp.commit_token(t)
        return ok
