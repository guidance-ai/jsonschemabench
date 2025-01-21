from .engine import Engine
from huggingface_hub import hf_hub_download

import llama_cpp
import ctypes
import llama_cpp.llama_grammar
import llama_cpp._internals
import numpy as np
import logging

from .json_schema import json_schema_to_gbnf


class LlamaCppEngine(Engine):
    def __init__(self):
        super().__init__()

    def init(self):
        llama_cpp.llama_backend_init()

        logging.getLogger("llama-cpp-python").setLevel(logging.ERROR)

        assert "Llama-3" in self.tokenizer_model_id
        model_path = hf_hub_download(
            repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
            filename="Llama-3.2-1B-Instruct-Q6_K.gguf",
        )

        params = llama_cpp.llama_model_default_params()
        model = llama_cpp.llama_load_model_from_file(model_path.encode("utf-8"), params)
        assert model
        self.model = model

        n_vocab = llama_cpp.llama_n_vocab(self.model)
        token_data_arr = llama_cpp._internals.LlamaTokenDataArray(n_vocab=n_vocab)
        token_data_arr.candidates_data.id[:] = token_data_arr.default_candidates_data_id
        token_data_arr.candidates_data.logit[:] = 0.0
        token_data_arr.candidates_data.p[:] = token_data_arr.default_candidates_data_p
        self.token_data_arr = token_data_arr

    def get_id(self):
        return "llamacpp"

    def get_name(self):
        return "llama.cpp"

    def get_module(self):
        return "llama-cpp-python"

    def compile_grammar(self, schema: dict):
        grm = json_schema_to_gbnf(schema)
        if self.debug:
            self.log_single(grm)
        self.sampler = llama_cpp.llama_sampler_init_grammar(
            self.model, grm.encode("utf-8"), b"root"
        )

    def reset(self):
        llama_cpp.llama_sampler_reset(self.sampler)

    def compute_mask(self):
        self.token_data_arr.candidates_data.logit[:] = 0.0
        p = ctypes.byref(self.token_data_arr.candidates)
        llama_cpp.llama_sampler_apply(self.sampler, p)

    def commit_token(self, t: int) -> bool:
        ok = self.token_data_arr.candidates_data.logit[t] >= 0
        if self.debug:
            self.dump_mask()
        if ok:
            llama_cpp.llama_sampler_accept(self.sampler, t)
        return ok

    def dump_mask(self):
        set = np.where(self.token_data_arr.candidates_data.logit >= 0.0)
        tok_list = list(set[0])
        r = f"Tokens {len(tok_list)}: " + ", ".join(
            [self.tok_to_str(t) for t in tok_list[0:100]]
        )
        print(r)

    def tok_to_str(self, tok: int):
        return repr(self.tok_to_bytes(tok))

    def tok_to_bytes(self, tok: int):
        size = 256
        buf = ctypes.create_string_buffer(size)
        special = True
        l = llama_cpp.llama_token_to_piece(self.model, tok, buf, size, 0, special)
        return bytes(buf[0:l])
