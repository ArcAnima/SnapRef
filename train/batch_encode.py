"""Batched Anima text encoding (close to, NOT bit-identical with, anima_env.encode_prompt).

    from batch_encode import BatchEncoder
    enc = BatchEncoder(clip, dm, device); ctxs = enc(list_of_texts)   # -> list of [tokens, 1024] bf16 (CPU)

encode_prompt runs Qwen3-0.6B and the llm_adapter on one prompt at a time (launch-overhead bound: ~33 prompts/s
per process on an H100). Here prompts are grouped by exact (Qwen3 length, T5 length), so batches need no padding
and no masks. Measured on pilot prompts (2026-10-30): 146 prompts/s (4.4x); a batch of more than one row takes
other bf16 GEMM kernels, so outputs differ from encode_prompt by up to 0.04 abs (values up to ~1.3; padded
batches with masks: ~1.3% relative L2, cos >= 0.9998). Whether that matters for training is not measured yet.
Prompts with non-1.0 token weights (e.g. "(tag:1.2)") go through encode_prompt one by one.
`self_check(texts)` returns the max abs difference against encode_prompt.
"""
from __future__ import annotations

import collections

import torch

import anima_env


class BatchEncoder:
    def __init__(self, clip, dm, device, batch: int = 128):
        self.clip, self.dm, self.device, self.batch = clip, dm, device, batch
        self.te = clip.cond_stage_model   # AnimaTEModel (SD1ClipModel wrapper around Qwen3_06BModel)
        self.qwen = getattr(self.te, self.te.clip)
        self._ready = False

    def _prepare(self, tokens):
        # what encode_from_tokens_scheduled does before encoding: load the weights, set the execution device
        self.clip.load_model(tokens)
        self.te.reset_clip_options()
        self.te.set_clip_options({"execution_device": self.clip.patcher.load_device})
        self._ready = True

    @torch.inference_mode()
    def _encode_same_length(self, toks: list[dict]) -> list[torch.Tensor]:
        qwen_ids = [[k[0] for k in tk["qwen3_06b"][0]] for tk in toks]
        z = self.qwen(qwen_ids)[0].to(self.device, torch.bfloat16)        # [B, Lq, 1024], no padding
        ids = torch.tensor([[k[0] for k in tk["t5xxl"][0]] for tk in toks], device=self.device)
        w = torch.tensor([[k[1] for k in tk["t5xxl"][0]] for tk in toks], dtype=torch.bfloat16, device=self.device)
        out = self.dm.llm_adapter(z, ids) * w.unsqueeze(-1)                 # [B, Lt, 1024], no padding
        return list(out.float().cpu().to(torch.bfloat16).unbind(0))

    def __call__(self, texts: list[str]) -> list[torch.Tensor]:
        res: list[torch.Tensor | None] = [None] * len(texts)
        groups = collections.defaultdict(list)
        toks = {}
        for i, t in enumerate(texts):
            tk = self.clip.tokenize(t)
            if any(k[1] != 1.0 for k in tk["t5xxl"][0]):
                res[i] = anima_env.encode_prompt(self.clip, self.dm, t, self.device)
                self._ready = False   # encode_prompt resets the clip options
                continue
            toks[i] = tk
            groups[(len(tk["qwen3_06b"][0]), len(tk["t5xxl"][0]))].append(i)
        for idx in groups.values():
            for s in range(0, len(idx), self.batch):
                part = idx[s: s + self.batch]
                if not self._ready:
                    self._prepare(toks[part[0]])
                for i, c in zip(part, self._encode_same_length([toks[i] for i in part])):
                    res[i] = c
        return res

    def self_check(self, texts: list[str]) -> float:
        ref = [anima_env.encode_prompt(self.clip, self.dm, t, self.device).float() for t in texts]
        self._ready = False
        got = [c.float() for c in self(texts)]
        assert all(a.shape == b.shape for a, b in zip(ref, got)), [(a.shape, b.shape) for a, b in zip(ref, got)][:3]
        return max(float((a - b).abs().max()) for a, b in zip(ref, got))
