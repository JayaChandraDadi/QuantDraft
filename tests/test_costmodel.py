import json

import numpy as np
import pytest

from quantdraft.controllers import CostModelGamma, FixedGamma, HeuristicGamma, make_controller
from quantdraft.costmodel import (
    CostModel,
    expected_tokens,
    fit_alpha_from_lengths,
    optimal_gamma,
    speedup,
)


def test_expected_tokens_limits():
    assert expected_tokens(0.0, 5) == 1.0
    assert expected_tokens(1.0, 5) == 6.0
    assert expected_tokens(0.5, 0) == 1.0
    assert expected_tokens(0.8, 4) == pytest.approx(sum(0.8**i for i in range(5)))


def test_expected_tokens_matches_simulation():
    rng = np.random.default_rng(0)
    alpha, gamma = 0.7, 4
    n = []
    for _ in range(20000):
        k = 0
        while k < gamma and rng.random() < alpha:
            k += 1
        n.append(k + 1)
    assert np.mean(n) == pytest.approx(expected_tokens(alpha, gamma), rel=0.02)


def test_speedup_formula():
    # perfect, free draft: speedup = gamma + 1
    assert speedup(1.0, 4, 0.0, 1.0) == pytest.approx(5.0)
    # useless draft: always slower
    assert speedup(0.0, 4, 0.2, 1.0) < 1.0


def test_optimal_gamma_behaviour():
    assert optimal_gamma(0.0, 0.1) == 0
    gs = [optimal_gamma(a, 0.1) for a in (0.5, 0.7, 0.85, 0.95)]
    assert gs == sorted(gs) and gs[-1] > gs[0]
    assert optimal_gamma(0.9, 0.05) >= optimal_gamma(0.9, 0.5)


def test_costmodel_interpolation_and_roundtrip():
    cm = CostModel(t_draft=0.1, t_target={1: 1.0, 3: 1.2, 5: 1.4})
    assert cm.target_cost(4) == pytest.approx(1.3)
    assert cm.target_cost(1) == 1.0
    cm2 = CostModel.from_dict(json.loads(json.dumps(cm.to_dict())))
    assert cm2.t_target == cm.t_target and cm2.t_draft == cm.t_draft


def test_fit_alpha_from_lengths():
    rng = np.random.default_rng(1)
    alpha, gamma = 0.6, 5
    acc = []
    for _ in range(5000):
        k = 0
        while k < gamma and rng.random() < alpha:
            k += 1
        acc.append(k)
    assert fit_alpha_from_lengths(acc, [gamma] * len(acc)) == pytest.approx(alpha, abs=0.02)


def test_heuristic_schedule():
    h = HeuristicGamma(start=4, min_gamma=1, max_gamma=8)
    h.update(4, 4, [])
    assert h.next_gamma() == 6
    h.update(6, 6, [])
    h.update(8, 8, [])
    assert h.next_gamma() == 8
    for _ in range(20):
        h.update(h.next_gamma(), 0, [])
    assert h.next_gamma() == 1


def test_costmodel_controller_falls_back_to_plain_decoding():
    cm = CostModel(t_draft=0.9, t_target={1: 1.0, 2: 1.05})
    c = CostModelGamma(cost=cm, alpha0=0.3, warmup=0, probe_every=1000)
    assert c.next_gamma() == 0


def test_costmodel_controller_goes_long_with_cheap_good_draft():
    cm = CostModel(t_draft=0.05, t_target={1: 1.0, 2: 1.0})
    c = CostModelGamma(cost=cm, alpha0=0.95, warmup=0)
    assert c.next_gamma() >= 6


def test_costmodel_controller_tracks_alpha():
    cm = CostModel(t_draft=0.1, t_target={1: 1.0})
    c = CostModelGamma(cost=cm, alpha0=0.9, warmup=0, ema=0.5)
    hi = c.next_gamma()
    for _ in range(20):
        c.update(hi, 0, [0.1])
    assert c.alpha < 0.2 and c.next_gamma() < hi


def test_controller_learns_costs_online():
    c = CostModelGamma(start=3, warmup=2)
    assert c.next_gamma() == 3
    c.update(3, 2, [1, 1, 0], t_draft=0.3, t_verify=1.1)
    assert c.next_gamma() == 0
    c.update(0, 0, [], t_draft=0.0, t_verify=1.0)
    assert c.cost.ready
    assert c.cost.t_target[1] == 1.0 and c.cost.t_target[4] == 1.1


def test_make_controller():
    assert isinstance(make_controller("4"), FixedGamma)
    assert isinstance(make_controller("heuristic"), HeuristicGamma)
    assert isinstance(make_controller("auto"), CostModelGamma)
