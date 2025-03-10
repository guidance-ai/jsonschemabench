from .engine import Engine
import json
from outlines_core import Vocabulary, Guide, Index
from outlines_core.json_schema import build_regex_from_schema


class OutlinesEngine(Engine):
    def __init__(self):
        super().__init__()

    def get_id(self):
        return "outlines"

    def get_name(self):
        return "Outlines"

    def init(self):
        self.vocabulary = Vocabulary.from_pretrained(self.tokenizer.name_or_path)

    def compile_grammar(self, schema: dict):
        rx = build_regex_from_schema(json.dumps(schema))
        self.index = Index(rx, self.vocabulary)
        self.guide = Guide(self.index)

    def reset(self):
        self.guide  = Guide(self.index)

    def compute_mask(self):
        self.res = self.guide.get_tokens()

    def commit_token(self, t: int) -> bool:
        ok = t in self.res
        if ok:
            self.guide.advance(t)
        return ok
