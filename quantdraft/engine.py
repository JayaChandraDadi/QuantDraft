"""Lossless speculative decoding for mlx-lm models.

Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative
Decoding* (ICML 2023): a cheap draft model proposes ``gamma`` tokens, the
target model scores all of them in one forward pass, and a rejection-sampling
rule keeps a prefix so that the output is distributed *exactly* as if the
target had generated alone.

Verification modes
------------------
``rejection`` (default)
    The paper's rule. Draft token ``d ~ q`` is accepted with probability
    ``min(1, p(d) / q(d))``; on rejection the replacement is sampled from
    ``norm(max(0, p - q))``. Per-position acceptance probability is
    ``sum_v min(p_v, q_v) = 1 - TV(p, q)``.
``match``
    What ``mlx_lm.generate --draft-model`` does: sample the target
    independently and accept while the two samples agree. Also lossless, but
    the acceptance probability is only ``sum_v p_v q_v`` -- strictly lower
    whenever ``temperature > 0`` and the distributions are not one-hot.

With ``temperature == 0`` both reduce to greedy argmax matching, and the
output is token-for-token identical to greedy decoding with the target.

KV-cache bookkeeping
--------------------
Invariant at the start of every round: each model's cache holds every
committed token except a short ``pending`` list that the model has not yet
processed. The target's pending list is always the single last token ``y``.
The draft's is ``[y]`` -- or ``[d_gamma, y]`` after a fully accepted round,
because the last draft token was sampled but never fed back through the draft.
After verification both caches are trimmed back to the accepted prefix.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache

from .controllers import FixedGamma, GammaController


@dataclass
class GenerationStats:
    """Everything needed to analyse a run after the fact."""

    prompt_tokens: int = 0
    generated: int = 0
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0
    gammas: list[int] = field(default_factory=list)
    accepted_per_round: list[int] = field(default_factory=list)
    # Per tested position: probability the rule would accept the draft token
    # (sum_v min(p, q) for rejection; sum_v p*q for match; 0/1 when greedy).
    alpha_terms: list[float] = field(default_factory=list)
    t_prefill: float = 0.0
    t_draft: float = 0.0
    t_verify: float = 0.0
    t_total: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens that were accepted."""
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def alpha(self) -> float:
        """Mean per-position acceptance probability (Leviathan's alpha)."""
        return sum(self.alpha_terms) / len(self.alpha_terms) if self.alpha_terms else 0.0

    @property
    def tokens_per_round(self) -> float:
        """Generated tokens per target forward pass (1.0 for plain decoding)."""
        return self.generated / self.rounds if self.rounds else 0.0

    @property
    def decode_tps(self) -> float:
        t = self.t_total - self.t_prefill
        return self.generated / t if t > 0 else 0.0

    def summary(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "generated": self.generated,
            "rounds": self.rounds,
            "drafted": self.drafted,
            "accepted": self.accepted,
            "acceptance_rate": self.acceptance_rate,
            "alpha": self.alpha,
            "tokens_per_round": self.tokens_per_round,
            "mean_gamma": sum(self.gammas) / len(self.gammas) if self.gammas else 0.0,
            "target_forwards": self.target_forwards,
            "draft_forwards": self.draft_forwards,
            "t_prefill": self.t_prefill,
            "t_draft": self.t_draft,
            "t_verify": self.t_verify,
            "t_total": self.t_total,
            "decode_tps": self.decode_tps,
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _check_cache(cache, name):
    if not can_trim_prompt_cache(cache):
        raise ValueError(
            f"{name} uses a KV cache that cannot be trimmed (e.g. a rotating "
            "sliding-window cache); speculative decoding needs rollback"
        )


def _prefill(model: nn.Module, cache, tokens: Sequence[int], step: int = 512) -> None:
    for i in range(0, len(tokens), step):
        model(mx.array(tokens[i : i + step])[None], cache=cache)
        mx.eval([c.state for c in cache])


def _last_logits(model: nn.Module, cache, tokens: Sequence[int] | mx.array) -> mx.array:
    x = tokens if isinstance(tokens, mx.array) else mx.array(list(tokens))
    return model(x.reshape(1, -1), cache=cache)[0, -1].astype(mx.float32)


def _sample(logits: mx.array, temperature: float) -> mx.array:
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    return mx.random.categorical(logits / temperature)


def _leading_true(mask: mx.array) -> mx.array:
    """Number of leading True values of a 1-D boolean array."""
    return mx.cumprod(mask.astype(mx.int32)).sum()


# --------------------------------------------------------------------------- #
# verification rules (pure functions -- tested directly)
# --------------------------------------------------------------------------- #
def verify_greedy(target_logits: mx.array, draft_tokens: mx.array):
    """Accept the longest prefix where draft == target argmax.

    ``target_logits`` has ``gamma + 1`` rows (one per drafted position plus
    the bonus position). Returns ``(n_accepted, next_token, alpha_terms)``.
    """
    g = draft_tokens.size
    tgt = mx.argmax(target_logits, axis=-1)
    match = tgt[:g] == draft_tokens if g else mx.array([], dtype=mx.bool_)
    n = _leading_true(match) if g else mx.array(0)
    n_i = int(n.item())
    x = int(tgt[n_i].item())
    tested = min(n_i + 1, g)
    alpha = match[:tested].astype(mx.float32).tolist() if g else []
    return n_i, x, alpha


def verify_rejection(
    p: mx.array, q: mx.array | None, draft_tokens: mx.array, u: mx.array | None = None
):
    """Leviathan et al.'s rejection rule.

    ``p``: target probabilities, ``(gamma + 1, V)``. ``q``: draft
    probabilities the draft tokens were sampled from, ``(gamma, V)``.
    ``u``: optional uniforms (for tests); drawn from ``mx.random`` otherwise.
    """
    g = draft_tokens.size
    if g == 0:
        x = int(mx.random.categorical(mx.log(p[0])).item())
        return 0, x, []
    idx = draft_tokens.reshape(-1, 1)
    p_d = mx.take_along_axis(p[:g], idx, axis=1)[:, 0]
    q_d = mx.take_along_axis(q, idx, axis=1)[:, 0]
    if u is None:
        u = mx.random.uniform(shape=(g,))
    # accept iff u < p/q, written without the division (q_d > 0: d was sampled from q)
    accept = u * q_d < p_d
    n = _leading_true(accept)
    overlap = mx.minimum(p[:g], q).sum(axis=-1)
    n_i = int(n.item())
    if n_i < g:
        resid = mx.maximum(p[n_i] - q[n_i], 0.0)
        mass = resid.sum()
        dist = mx.where(mass > 0, resid / mx.maximum(mass, 1e-30), p[n_i])
    else:
        dist = p[g]
    x = int(mx.random.categorical(mx.log(dist)).item())
    tested = min(n_i + 1, g)
    return n_i, x, overlap[:tested].tolist()


def verify_match(p_logits: mx.array, draft_tokens: mx.array, temperature: float,
                 p: mx.array | None = None, q: mx.array | None = None):
    """mlx-lm's rule: sample the target independently, accept while equal."""
    g = draft_tokens.size
    samples = mx.random.categorical(p_logits / temperature)
    match = samples[:g] == draft_tokens if g else mx.array([], dtype=mx.bool_)
    n_i = int(_leading_true(match).item()) if g else 0
    x = int(samples[n_i].item())
    tested = min(n_i + 1, g)
    alpha = []
    if g and p is not None and q is not None:
        alpha = (p[:g] * q).sum(axis=-1)[:tested].tolist()
    return n_i, x, alpha


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def autoregressive_generate(
    model: nn.Module,
    prompt: Sequence[int],
    max_tokens: int,
    temperature: float = 0.0,
    seed: int = 0,
    eos_ids: Iterable[int] = (),
):
    """Plain one-token-at-a-time decoding (the baseline), same bookkeeping."""
    eos = set(eos_ids)
    mx.random.seed(seed)
    stats = GenerationStats(prompt_tokens=len(prompt))
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    _prefill(model, cache, prompt[:-1])
    stats.t_prefill = time.perf_counter() - t0
    y = mx.array([prompt[-1]])
    out: list[int] = []
    while len(out) < max_tokens:
        tok = _sample(_last_logits(model, cache, y), temperature)
        y = tok.reshape(1)
        out.append(int(tok.item()))
        stats.rounds += 1
        stats.target_forwards += 1
        if out[-1] in eos:
            break
    stats.generated = len(out)
    stats.t_total = time.perf_counter() - t0
    stats.t_verify = stats.t_total - stats.t_prefill
    return out, stats


def speculative_generate(
    target: nn.Module,
    draft: nn.Module,
    prompt: Sequence[int],
    max_tokens: int,
    gamma: int | GammaController = 4,
    temperature: float = 0.0,
    verify: str = "rejection",
    seed: int = 0,
    eos_ids: Iterable[int] = (),
):
    """Speculative decoding. Returns ``(tokens, GenerationStats)``.

    ``gamma`` is either a fixed draft length or a :class:`GammaController`
    that chooses it every round.
    """
    if verify not in ("rejection", "match"):
        raise ValueError("verify must be 'rejection' or 'match'")
    if len(prompt) < 1:
        raise ValueError("prompt must contain at least one token")
    controller = gamma if isinstance(gamma, GammaController) else FixedGamma(int(gamma))
    greedy = temperature == 0
    eos = set(eos_ids)
    mx.random.seed(seed)
    stats = GenerationStats(prompt_tokens=len(prompt))

    t_cache = make_prompt_cache(target)
    d_cache = make_prompt_cache(draft)
    _check_cache(t_cache, "target")
    _check_cache(d_cache, "draft")

    t0 = time.perf_counter()
    _prefill(target, t_cache, prompt[:-1])
    _prefill(draft, d_cache, prompt[:-1])
    stats.t_prefill = time.perf_counter() - t0

    y = prompt[-1]  # last committed token, not yet in either cache
    d_pending = [y]
    out: list[int] = []

    while len(out) < max_tokens:
        g = max(0, min(controller.next_gamma(), max_tokens - len(out) - 1))

        # ---- draft: g autoregressive steps ------------------------------
        td = time.perf_counter()
        d_toks, q_rows = [], []
        feed = mx.array(d_pending)
        for _ in range(g):
            logits = _last_logits(draft, d_cache, feed)
            if greedy:
                tok = mx.argmax(logits)
            else:
                scaled = logits / temperature
                q_rows.append(mx.softmax(scaled))
                tok = mx.random.categorical(scaled)
            d_toks.append(tok)
            feed = tok.reshape(1)
            mx.async_eval(tok)
        d_arr = mx.stack(d_toks) if g else mx.array([], dtype=mx.uint32)
        mx.eval(d_arr)
        dt_draft = time.perf_counter() - td
        stats.t_draft += dt_draft
        stats.draft_forwards += g

        # ---- verify: one target pass over [y, d_1..d_g] -------------------
        tv = time.perf_counter()
        v_in = mx.concatenate([mx.array([y], dtype=d_arr.dtype), d_arr]) if g else mx.array([y])
        logits_t = target(v_in.reshape(1, -1), cache=t_cache)[0].astype(mx.float32)
        if greedy:
            n, x, alpha = verify_greedy(logits_t, d_arr)
        elif verify == "rejection":
            p = mx.softmax(logits_t / temperature, axis=-1)
            q = mx.stack(q_rows) if g else None
            n, x, alpha = verify_rejection(p, q, d_arr)
        else:
            p = mx.softmax(logits_t / temperature, axis=-1) if g else None
            q = mx.stack(q_rows) if g else None
            n, x, alpha = verify_match(logits_t, d_arr, temperature, p, q)
        dt_verify = time.perf_counter() - tv
        stats.t_verify += dt_verify
        stats.target_forwards += 1

        # ---- commit and roll back the caches ------------------------------
        drafted = d_arr.tolist() if g else []
        emitted = drafted[:n] + [x]
        stop = False
        for i, t in enumerate(emitted):
            if t in eos:
                emitted = emitted[: i + 1]
                stop = True
                break
        out.extend(emitted)

        trim_prompt_cache(t_cache, g - n)  # keep y, d_1..d_n
        if g == 0:
            d_pending = d_pending + [x]
        elif n == g:
            d_pending = [drafted[-1], x]  # d_g was never fed to the draft
        else:
            trim_prompt_cache(d_cache, g - 1 - n)
            d_pending = [x]
        y = x

        stats.rounds += 1
        stats.drafted += g
        stats.accepted += n
        stats.gammas.append(g)
        stats.accepted_per_round.append(n)
        stats.alpha_terms.extend(alpha)
        controller.update(g, n, alpha, dt_draft, dt_verify)
        if stop:
            break

    out = out[:max_tokens]
    stats.generated = len(out)
    stats.t_total = time.perf_counter() - t0
    return out, stats
