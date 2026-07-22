"""GRID engine for guidance-ai/jsonschemabench MaskBench.

Drop this file into `maskbench/maskbench/` and register it in `runner.py`
(see accompanying README). Requires `pip install grid-guardrail`.

GRID compiles the JSON Schema to an LALR(1) grammar with constrained
terminals (grid.jsonschema); masks come from the configuration-keyed
viable-prefix walk. Unenforced constraints are RECORDED per schema (default
mode) or DECLARED (strict mode) — never silent; see
grid/jsonschema/SUPPORT.md for the keyword matrix.
"""

from .engine import Engine


class GridEngine(Engine):

    def __init__(self, strict: bool = False):
        super().__init__()
        self.strict = strict

    def init(self):
        from grid.models.hf_adapter import HFTokenizerAdapter
        from grid.trie.build import build_trie

        self.adapter = HFTokenizerAdapter(self.tokenizer)
        # per-tokenizer token trie, shared across schemas (like llg/xgr init)
        self.trie = build_trie(self.adapter)

    def get_id(self):
        return "grid"

    def get_name(self):
        return "GRID" + (" (strict)" if self.strict else "")

    def get_module(self):
        return "grid-guardrail"

    def compile_grammar(self, schema: dict):
        from grid.grammar import spec
        from grid.grammar.projection import RoleProjection
        from grid.guide import GridGuide
        from grid.jsonschema import compile_json_schema
        from grid.lalr.compile import compile_tables
        from grid.lexer.dfa import build_scanner

        src, recorded = compile_json_schema(schema, strict=self.strict)
        grammar = spec.load(src)
        proj = RoleProjection.full(grammar).build()
        tables = compile_tables(proj)
        dfa = build_scanner(grammar.terminals, grammar.terminal_order)
        self.guide = GridGuide(tables=tables, dfa=dfa, trie=self.trie,
                               adapter=self.adapter)
        self.recorded = sorted(recorded)
        if recorded:
            self.log_single(f"recorded (unenforced): {self.recorded}")

    def reset(self):
        self.state = self.guide.initial_state

    def compute_mask(self):
        self.mask, _ = self.guide._mask_ids(self.state)

    def commit_token(self, t: int) -> bool:
        ok = bool((self.mask == t).any())
        if ok:
            self.state = self.guide.get_next_state(self.state, t)
        return ok
