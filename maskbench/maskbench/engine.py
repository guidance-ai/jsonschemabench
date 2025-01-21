from transformers import PreTrainedTokenizer

class Engine:
    tokenizer: PreTrainedTokenizer
    tokenizer_model_id: str

    def __init__(self):
        pass

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
