"""Analytical speedup model for speculative decoding.

With per-position acceptance probability ``alpha`` (assumed i.i.d.) and draft
length ``gamma``, one round emits on average

    E[tokens | alpha, gamma] = (1 - alpha^(gamma+1)) / (1 - alpha)

tokens (Leviathan et al., Eq. 1). A round costs ``gamma`` draft forwards plus
one target forward over ``gamma + 1`` tokens. On memory-bound hardware the
latter is almost the cost of a single-token forward, which is the whole point.

    speedup = E[tokens] * t_target(1) / (gamma * t_draft + t_target(gamma + 1))
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def expected_tokens(alpha: float, gamma: int) -> float:
    if gamma <= 0:
        return 1.0
    if alpha >= 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def speedup(alpha: float, gamma: int, t_draft: float, t_target_1: float,
            t_target_k: float | None = None) -> float:
    """Expected wall-clock speedup over plain decoding with the target.

    ``t_target_k`` is the cost of a target pass over ``gamma + 1`` tokens;
    defaults to ``t_target_1`` (perfectly memory-bound verification).
    """
    t_k = t_target_1 if t_target_k is None else t_target_k
    return expected_tokens(alpha, gamma) * t_target_1 / (gamma * t_draft + t_k)


def bandwidth_cost_ratio(draft_bytes: float, target_bytes: float) -> float:
    """First-order draft/target cost ratio for memory-bound decoding: the time
    of a forward pass is ~ the bytes of weights it streams."""
    return draft_bytes / target_bytes


@dataclass
class CostModel:
    """Per-forward costs, either profiled up front or learned online.

    ``t_target[k]`` is the cost of a target forward over ``k`` tokens.
    Unseen ``k`` are filled by a least-squares line ``a + b*k`` over the
    observed ones (verification cost is close to affine in ``k`` for small
    ``k``).
    """

    t_draft: float | None = None
    t_target: dict[int, float] = field(default_factory=dict)
    ema: float = 0.8

    def observe_draft(self, seconds_per_token: float) -> None:
        self.t_draft = seconds_per_token if self.t_draft is None else (
            self.ema * self.t_draft + (1 - self.ema) * seconds_per_token)

    def observe_target(self, k: int, seconds: float) -> None:
        old = self.t_target.get(k)
        self.t_target[k] = seconds if old is None else self.ema * old + (1 - self.ema) * seconds

    def target_cost(self, k: int) -> float | None:
        if k in self.t_target:
            return self.t_target[k]
        if not self.t_target:
            return None
        ks = np.array(sorted(self.t_target))
        ts = np.array([self.t_target[int(i)] for i in ks])
        if len(ks) == 1:
            return float(ts[0])
        b, a = np.polyfit(ks, ts, 1)
        return float(max(a + b * k, ts.min()))

    @property
    def ready(self) -> bool:
        return self.t_draft is not None and self.target_cost(1) is not None

    def speedup(self, alpha: float, gamma: int) -> float:
        return speedup(alpha, gamma, self.t_draft, self.target_cost(1),
                       self.target_cost(gamma + 1))

    def best_gamma(self, alpha: float, max_gamma: int) -> int:
        """Argmax over gamma in [0, max_gamma] of the expected speedup
        (gamma = 0 means plain decoding is better)."""
        best, best_s = 0, 1.0
        for g in range(1, max_gamma + 1):
            s = self.speedup(alpha, g)
            if s > best_s:
                best, best_s = g, s
        return best

    def to_dict(self) -> dict:
        return {"t_draft": self.t_draft, "t_target": {str(k): v for k, v in self.t_target.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "CostModel":
        return cls(t_draft=d["t_draft"], t_target={int(k): v for k, v in d["t_target"].items()})


def optimal_gamma(alpha: float, cost_ratio: float, max_gamma: int = 16) -> int:
    """Best fixed gamma when verification is perfectly memory-bound and the
    draft costs ``cost_ratio`` of a target forward."""
    cm = CostModel(t_draft=cost_ratio, t_target={1: 1.0})
    return cm.best_gamma(alpha, max_gamma)


def fit_alpha_from_lengths(accepted: list[int], gammas: list[int]) -> float:
    """Maximum-likelihood alpha from per-round accepted counts under the i.i.d.
    model (truncated geometric). Useful when only token counts are logged."""
    acc = np.asarray(accepted)
    gam = np.asarray(gammas)
    mask = gam > 0
    acc, gam = acc[mask], gam[mask]
    if acc.size == 0:
        return 0.0
    # successes = accepted tokens; failures = rounds that stopped early
    successes = acc.sum()
    failures = (acc < gam).sum()
    return float(successes / (successes + failures)) if successes + failures else 0.0
