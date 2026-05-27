from .engine import Engine

import llguidance as llg
from llguidance.numpy import fill_next_token_bitmask, allocate_token_bitmask
import llguidance.hf
import ctypes
import json
import time


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
        word = int(self.mask_data[0, t >> 5]) & 0xFFFFFFFF
        ok = (word & (1 << (t & 31))) != 0
        if ok:
            self.interp.consume_token(t)
        elif self.interp.is_error():
            raise ValueError(self.interp.get_error())
        return ok


class LlgRawBytesEngine(LlgEngine):
    def get_id(self):
        return "llg-rawbytes"

    def get_name(self):
        return "LLGuidanceRawBytes"

    def init(self):
        self.llg_tokenizer = llguidance.hf.from_tokenizer(self.tokenizer)
        mask_words = (self.llg_tokenizer.vocab_size + 31) // 32
        self.mask_buf = bytearray(mask_words * 4)
        self.mask_ptr = ctypes.addressof(ctypes.c_char.from_buffer(self.mask_buf))

    def compute_mask(self):
        self.interp.unsafe_compute_mask_ptr(self.mask_ptr, len(self.mask_buf))

    def commit_token(self, t: int) -> bool:
        ok = (self.mask_buf[t >> 3] & (1 << (t & 7))) != 0
        if ok:
            self.interp.consume_token(t)
        elif self.interp.is_error():
            raise ValueError(self.interp.get_error())
        return ok


class LlgFastForwardEngine(LlgRawBytesEngine):
    def get_id(self):
        return "llg-ff"

    def get_name(self):
        return "LLGuidanceFastForward"

    def process_tokens(self, tokens: list[int]) -> dict:
        idx = 0
        all_step_us = []
        total_us = 0
        max_step_us = 0
        mask_calls = 0
        mask_call_us = 0
        ff_tokens = 0
        ff_us = 0

        def add_step(elapsed_us: int, count: int = 1):
            nonlocal total_us, max_step_us
            if count <= 0:
                return
            per_token_us = max(1, elapsed_us // count)
            all_step_us.extend([per_token_us] * count)
            total_us += per_token_us * count
            max_step_us = max(max_step_us, per_token_us)

        while idx < len(tokens):
            t0 = time.monotonic()
            forced = self.interp.compute_ff_tokens()
            if forced:
                remaining = len(tokens) - idx
                forced_len = len(forced)
                if forced_len > remaining or tokens[idx : idx + forced_len] != forced:
                    elapsed_us = int((time.monotonic() - t0) * 1_000_000)
                    add_step(elapsed_us)
                    return {
                        "accepted": False,
                        "all_mask_us": all_step_us,
                        "masks_us": total_us,
                        "max_mask_us": max_step_us,
                        "num_tokens": idx + 1,
                        "num_mask_calls": mask_calls,
                        "mask_call_us": mask_call_us,
                        "num_ff_tokens": ff_tokens,
                        "ff_us": ff_us + elapsed_us,
                    }
                self.interp.consume_tokens(forced)
                elapsed_us = int((time.monotonic() - t0) * 1_000_000)
                ff_tokens += forced_len
                ff_us += elapsed_us
                add_step(elapsed_us, forced_len)
                idx += forced_len
                continue

            self.compute_mask()
            ok = self.commit_token(tokens[idx])
            elapsed_us = int((time.monotonic() - t0) * 1_000_000)
            mask_calls += 1
            mask_call_us += elapsed_us
            add_step(elapsed_us)
            idx += 1
            if not ok:
                return {
                    "accepted": False,
                    "all_mask_us": all_step_us,
                    "masks_us": total_us,
                    "max_mask_us": max_step_us,
                    "num_tokens": idx,
                    "num_mask_calls": mask_calls,
                    "mask_call_us": mask_call_us,
                    "num_ff_tokens": ff_tokens,
                    "ff_us": ff_us,
                }

        return {
            "accepted": True,
            "all_mask_us": all_step_us,
            "masks_us": total_us,
            "max_mask_us": max_step_us,
            "num_tokens": idx,
            "num_mask_calls": mask_calls,
            "mask_call_us": mask_call_us,
            "num_ff_tokens": ff_tokens,
            "ff_us": ff_us,
        }
