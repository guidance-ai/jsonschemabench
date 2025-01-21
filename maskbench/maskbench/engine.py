from transformers import PreTrainedTokenizer


class Engine:
    tokenizer: PreTrainedTokenizer
    tokenizer_model_id: str

    def __init__(self):
        self.multi = False
        self.debug = False

    def get_id(self):
        raise NotImplementedError()

    def get_name(self):
        return self.get_id()

    def get_module(self):
        return self.get_id()

    def get_version(self):
        from importlib.metadata import version

        return version(self.get_module())

    def init(self):
        raise NotImplementedError()

    def compile_grammar(self, schema: dict):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def compute_mask(self):
        raise NotImplementedError()

    def commit_token(self, token: int) -> bool:
        raise NotImplementedError()

    def log_single(self, s: str):
        import sys

        if not self.multi:
            print(s, file=sys.stderr)
