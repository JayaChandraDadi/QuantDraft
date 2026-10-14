"""How many tokens to draft each round.

* :class:`FixedGamma` -- the paper's setting and mlx-lm's (``--num-draft-tokens``).
* :class:`HeuristicGamma` -- grow by 2 after a fully accepted round, shrink by
  1 otherwise (the schedule Hugging Face transformers uses for assisted
  generation).
* :class:`CostModelGamma` -- track the acceptance probability with an EMA and
  measured forward costs with a :class:`~quantdraft.costmodel.CostModel`,
  then pick the gamma that maximises the *expected speedup* each round. When
  the draft stops paying for itself (low acceptance, or a draft that is not
  much cheaper than the target) it falls back to gamma = 0, i.e. plain
  decoding, instead of slowing generation down.
"""

from __future__ import annotations

from .costmodel import CostModel


class GammaController:
    def next_gamma(self) -> int:
        raise NotImplementedError

    def update(self, gamma: int, accepted: int, alpha_terms: list[float],
               t_draft: float | None = None, t_verify: float | None = None) -> None:
        pass

    def describe(self) -> str:
        return type(self).__name__


class FixedGamma(GammaController):
    def __init__(self, gamma: int):
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        self.gamma = gamma

    def next_gamma(self) -> int:
        return self.gamma

    def describe(self) -> str:
        return f"fixed-{self.gamma}"


class HeuristicGamma(GammaController):
    def __init__(self, start: int = 4, min_gamma: int = 1, max_gamma: int = 12):
        self.gamma, self.min, self.max = start, min_gamma, max_gamma

    def next_gamma(self) -> int:
        return self.gamma

    def update(self, gamma, accepted, alpha_terms, t_draft=None, t_verify=None):
        if gamma == 0:
            return
        if accepted == gamma:
            self.gamma = min(self.max, self.gamma + 2)
        else:
            self.gamma = max(self.min, self.gamma - 1)

    def describe(self) -> str:
        return "heuristic"


class CostModelGamma(GammaController):
    """Expected-speedup-maximizing gamma.

    Args:
        cost: a profiled :class:`CostModel`. If ``None`` costs are learned
            online from the engine's per-round timings (after ``warmup``
            rounds at ``start`` tokens).
        alpha0: prior acceptance probability.
        ema: smoothing for alpha (per tested position).
        max_gamma: upper bound.
        probe_every: when the model says gamma = 0, still draft one token every
            this many rounds so a recovering acceptance rate is noticed.
    """

    def __init__(self, cost: CostModel | None = None, alpha0: float = 0.7,
                 ema: float = 0.9, max_gamma: int = 12, start: int = 4,
                 warmup: int = 3, probe_every: int = 8):
        self.cost = cost if cost is not None else CostModel()
        self.learn_costs = cost is None
        self.alpha = alpha0
        self.ema = ema
        self.max_gamma = max_gamma
        self.start = start
        self.warmup = warmup
        self.probe_every = probe_every
        self.rounds = 0
        self.since_probe = 0
        self.history: list[tuple[int, float]] = []

    def next_gamma(self) -> int:
        if self.rounds < self.warmup or not self.cost.ready:
            # learn the cost of both 1-token and (start+1)-token target passes
            return self.start if self.rounds % 2 == 0 else 0
        g = self.cost.best_gamma(self.alpha, self.max_gamma)
        if g == 0:
            self.since_probe += 1
            if self.since_probe >= self.probe_every:
                self.since_probe = 0
                return 1
        return g

    def update(self, gamma, accepted, alpha_terms, t_draft=None, t_verify=None):
        self.rounds += 1
        for a in alpha_terms:
            self.alpha = self.ema * self.alpha + (1 - self.ema) * a
        if self.learn_costs:
            if gamma > 0 and t_draft is not None:
                self.cost.observe_draft(t_draft / gamma)
            if t_verify is not None:
                self.cost.observe_target(gamma + 1, t_verify)
        self.history.append((gamma, self.alpha))

    def describe(self) -> str:
        return "cost-model"


def make_controller(spec: str, cost: CostModel | None = None, max_gamma: int = 12) -> GammaController:
    """``"4"`` -> fixed 4, ``"heuristic"``, ``"auto"`` -> cost model."""
    if spec == "heuristic":
        return HeuristicGamma(max_gamma=max_gamma)
    if spec == "auto":
        return CostModelGamma(cost=cost, max_gamma=max_gamma)
    return FixedGamma(int(spec))
