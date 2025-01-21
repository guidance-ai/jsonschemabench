from transformers import PreTrainedTokenizer, AutoConfig
from .engine import Engine
import json
import xgrammar as xgr


class XgrEngine(Engine):
    def __init__(self):
        super().__init__()
        self.compliant = False

    def init(self):
        config = AutoConfig.from_pretrained(self.tokenizer_model_id)
        # This can be larger than tokenizer.vocab_size due to paddings
        full_vocab_size = config.vocab_size
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer, vocab_size=full_vocab_size
        )
        self.token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        self.xgr_compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=1)

    def compile_grammar(self, schema: dict):
        if self.compliant:
            xgr_strict = False
            xgr_any_whitespace = True
        else:
            xgr_strict = True
            xgr_any_whitespace = False

        schema_s = json.dumps(schema)
        self.compiled_grammar = self.xgr_compiler.compile_json_schema(
            schema_s, any_whitespace=xgr_any_whitespace, strict_mode=xgr_strict
        )
        # print(compiled_grammar.grammar, file=sys.stderr)

    def reset(self):
        self.matcher = xgr.GrammarMatcher(self.compiled_grammar)

    def compute_mask(self):
        self.matcher.fill_next_token_bitmask(self.token_bitmask)

    def commit_token(self, t: int) -> bool:
        # TODO: check in self.token_bitmask
        ok = self.matcher.accept_token(t)
        return ok
