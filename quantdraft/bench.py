"""Benchmarks: forward-cost profiling and end-to-end decoding comparisons."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from .controllers import make_controller
from .costmodel import CostModel
from .engine import GenerationStats, autoregressive_generate, speculative_generate


def _timed_forward(model, cache, tokens: mx.array) -> float:
    t = time.perf_counter()
    out = model(tokens.reshape(1, -1), cache=cache)
    mx.eval(out)
    return time.perf_counter() - t


def profile_costs(target: nn.Module, draft: nn.Module | None, context: Sequence[int],
                  max_k: int = 9, repeats: int = 7) -> CostModel:
    """Measure ``t_target(k)`` for k = 1..max_k and ``t_draft`` (1 token) at a
    realistic context length. Each measurement trims the cache back, so all
    are taken at the same position. Medians over ``repeats`` after a warmup."""
    cm = CostModel()
    ctx = list(context)

    def measure(model, k):
        cache = make_prompt_cache(model)
        model(mx.array(ctx)[None], cache=cache)
        mx.eval([c.state for c in cache])
        toks = mx.array(ctx[-k:] if len(ctx) >= k else (ctx * k)[:k])
        _timed_forward(model, cache, toks)  # warmup
        trim_prompt_cache(cache, k)
        ts = []
        for _ in range(repeats):
            ts.append(_timed_forward(model, cache, toks))
            trim_prompt_cache(cache, k)
        return statistics.median(ts)

    for k in range(1, max_k + 1):
        cm.t_target[k] = measure(target, k)
    if draft is not None:
        cm.t_draft = measure(draft, 1)
    return cm


@dataclass
class RunConfig:
    gamma: str = "4"  # int, "heuristic" or "auto"
    temperature: float = 0.0
    verify: str = "rejection"

    @property
    def name(self) -> str:
        return f"g={self.gamma},T={self.temperature},{self.verify}"


def _aggregate(stats: list[GenerationStats]) -> dict:
    agg = GenerationStats()
    for s in stats:
        for f in ("prompt_tokens", "generated", "rounds", "drafted", "accepted",
                  "target_forwards", "draft_forwards"):
            setattr(agg, f, getattr(agg, f) + getattr(s, f))
        for f in ("t_prefill", "t_draft", "t_verify", "t_total"):
            setattr(agg, f, getattr(agg, f) + getattr(s, f))
        agg.gammas += s.gammas
        agg.accepted_per_round += s.accepted_per_round
        agg.alpha_terms += s.alpha_terms
    return agg.summary()


def run_baseline(target, prompts, max_tokens, temperature, seed=0, eos_ids=()):
    outs, stats = [], []
    for i, p in enumerate(prompts):
        o, s = autoregressive_generate(target, p, max_tokens, temperature, seed + i, eos_ids)
        outs.append(o)
        stats.append(s)
    return outs, _aggregate(stats)


def run_speculative(target, draft, prompts, max_tokens, cfg: RunConfig, seed=0,
                    eos_ids=(), cost: CostModel | None = None, max_gamma: int = 12):
    outs, stats = [], []
    for i, p in enumerate(prompts):
        ctrl = make_controller(cfg.gamma, cost=cost, max_gamma=max_gamma)
        o, s = speculative_generate(target, draft, p, max_tokens, ctrl,
                                    cfg.temperature, cfg.verify, seed + i, eos_ids)
        outs.append(o)
        stats.append(s)
    return outs, _aggregate(stats)


def warmup(target, draft, prompt, n=8):
    autoregressive_generate(target, prompt, n)
    if draft is not None:
        speculative_generate(target, draft, prompt, n, 2)
