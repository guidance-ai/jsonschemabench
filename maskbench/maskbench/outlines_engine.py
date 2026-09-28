from .engine import Engine
import json
from outlines_core import Vocabulary, Guide, Index
from outlines_core.json_schema import build_regex_from_schema
from outlines_core.kernels.torch import allocate_token_bitmask, fill_next_token_bitmask


class OutlinesEngine(Engine):
    def __init__(self):
        super().__init__()

    def get_id(self):
        return "outlines"

    def get_module(self):
        return "outlines-core"
    
    def get_name(self):
        return "Outlines"

    def init(self):
        self.vocabulary = Vocabulary.from_pretrained(self.tokenizer.name_or_path)
        self.mask = allocate_token_bitmask(len(self.vocabulary))

    def compile_grammar(self, schema: dict):
        rx = build_regex_from_schema(json.dumps(schema))
        self.index = Index(rx, self.vocabulary)
        self.guide = Guide(self.index)

    def reset(self):
        self.guide  = Guide(self.index)

    def compute_mask(self):
        fill_next_token_bitmask(self.guide, self.mask)

    def commit_token(self, t: int) -> bool:
        ok = True
        if ok:
            self.guide.advance(t, False)
        return ok
