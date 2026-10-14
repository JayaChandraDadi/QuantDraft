"""Losslessness: the core promise of speculative decoding."""

from collections import Counter

import mlx.core as mx
import numpy as np
import pytest
from scipy import stats as sstats

from quantdraft.controllers import CostModelGamma, FixedGamma, HeuristicGamma
from quantdraft.drafts import nested_draft, rtn_draft
from quantdraft.engine import (
    autoregressive_generate,
    speculative_generate,
    verify_rejection,
)

from .conftest import tiny_llama

PROMPT = [3, 17, 42, 8, 5, 11]


@pytest.fixture(scope="module")
def models():
    target = tiny_llama(seed=0, logit_scale=8.0)
    drafts = {
        "rtn2": rtn_draft(target, 2),
        "rtn4": rtn_draft(target, 4),
        "unrelated": tiny_llama(seed=99, logit_scale=8.0),  # low acceptance
        "self": target,  # perfect draft
    }
    return target, drafts


# --------------------------------------------------------------------------- #
# greedy: token-for-token identical to the target
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("draft_name", ["rtn2", "rtn4", "unrelated", "self"])
@pytest.mark.parametrize("gamma", [0, 1, 2, 5, 9])
def test_greedy_identical_to_target(models, draft_name, gamma):
    target, drafts = models
    ref, _ = autoregressive_generate(target, PROMPT, 40)
    out, st = speculative_generate(target, drafts[draft_name], PROMPT, 40, gamma=gamma)
    assert out == ref
    assert st.generated == 40
    assert st.target_forwards == st.rounds


@pytest.mark.parametrize("ctrl", [lambda: HeuristicGamma(), lambda: CostModelGamma(),
                                  lambda: FixedGamma(3)])
def test_greedy_identical_with_adaptive_controllers(models, ctrl):
    target, drafts = models
    ref, _ = autoregressive_generate(target, PROMPT, 60)
    out, _ = speculative_generate(target, drafts["rtn4"], PROMPT, 60, gamma=ctrl())
    assert out == ref


def test_perfect_draft_accepts_everything(models):
    target, drafts = models
    _, st = speculative_generate(target, drafts["self"], PROMPT, 31, gamma=5)
    assert st.acceptance_rate == 1.0
    assert st.tokens_per_round == pytest.approx(31 / st.rounds)
    assert st.rounds == 6  # 5 rounds of 6 tokens, then a final round of 1


def test_long_generation_cache_rollback(models):
    """Hundreds of accept/reject/rollback cycles must keep the KV caches
    aligned (any drift would change the greedy output)."""
    target, drafts = models
    ref, _ = autoregressive_generate(target, PROMPT, 200)
    out, st = speculative_generate(target, drafts["rtn2"], PROMPT, 200, gamma=4)
    assert out == ref
    assert 0 < st.accepted < st.drafted  # both outcomes exercised


def test_single_token_prompt_and_max_tokens(models):
    target, drafts = models
    for n in (1, 2, 7):
        ref, _ = autoregressive_generate(target, [5], n)
        out, _ = speculative_generate(target, drafts["rtn4"], [5], n, gamma=6)
        assert out == ref and len(out) == n


def test_eos_stops_generation(models):
    target, drafts = models
    ref, _ = autoregressive_generate(target, PROMPT, 40)
    eos = ref[10]
    first = ref.index(eos)
    out, _ = speculative_generate(target, drafts["rtn4"], PROMPT, 40, gamma=4, eos_ids={eos})
    assert out == ref[: first + 1]


# --------------------------------------------------------------------------- #
# sampling: the output *distribution* must equal the target's
# --------------------------------------------------------------------------- #
def _chisq_pvalue(obs, exp):
    obs = np.asarray(obs, dtype=np.float64)
    exp = np.asarray(exp, dtype=np.float64)
    exp = exp * obs.sum() / exp.sum()
    chi2 = ((obs - exp) ** 2 / exp).sum()
    return sstats.chi2.sf(chi2, len(obs) - 1)


def _np_softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rejection_rule_single_position_exact(seed):
    """Enumerate the draft token: P(emit x) = q(x)min(1,p/q) + P(reject) r(x) = p(x)."""
    rng = np.random.default_rng(seed)
    V = 7
    p = _np_softmax(rng.normal(size=V) * 2)
    q = _np_softmax(rng.normal(size=V) * 2)
    accept = np.minimum(1, p / q)
    reject_mass = (q * (1 - accept)).sum()
    resid = np.maximum(p - q, 0)
    resid /= resid.sum()
    emitted = q * accept + reject_mass * resid
    assert np.allclose(emitted, p)
    assert np.isclose((q * accept).sum(), np.minimum(p, q).sum())  # alpha = 1 - TV


def test_verify_rejection_monte_carlo():
    rng = np.random.default_rng(0)
    V = 6
    p1 = _np_softmax(rng.normal(size=V) * 1.5)
    q1 = _np_softmax(rng.normal(size=V) * 1.5)
    p = mx.array(np.stack([p1, _np_softmax(rng.normal(size=V))]).astype(np.float32))
    q = mx.array(q1[None].astype(np.float32))
    mx.random.seed(0)
    counts = Counter()
    N = 6000
    for i in range(N):
        d = int(rng.choice(V, p=q1))
        n, x, _ = verify_rejection(p, q, mx.array([d]))
        counts[d if n == 1 else x] += 1
    obs = np.array([counts[v] for v in range(V)])
    assert _chisq_pvalue(obs, p1 * N) > 1e-3, (obs / N, p1)


def _exact_two_token_distribution(model, prompt, T):
    V = model.args.vocab_size
    l1 = model(mx.array([prompt]))[0, -1].astype(mx.float32)
    p1 = np.array(mx.softmax(l1 / T))
    joint = np.zeros((V, V))
    batch = mx.array([prompt + [a] for a in range(V)])
    l2 = model(batch)[:, -1].astype(mx.float32)
    p2 = np.array(mx.softmax(l2 / T, axis=-1))
    joint = p1[:, None] * p2
    return joint


@pytest.mark.parametrize("verify", ["rejection", "match"])
def test_full_engine_two_token_distribution(verify):
    """End to end (drafting, caches, verification): the empirical
    distribution of the first two generated tokens matches the target's exact
    joint distribution."""
    V = 8
    target = tiny_llama(vocab=V, hidden=64, seed=0, logit_scale=3.0)
    draft = tiny_llama(vocab=V, hidden=64, seed=5, logit_scale=3.0)
    prompt = [1, 4, 2]
    T = 1.0
    joint = _exact_two_token_distribution(target, prompt, T)
    N = 2500
    counts = Counter()
    for s in range(N):
        out, _ = speculative_generate(target, draft, prompt, 2, gamma=2,
                                      temperature=T, verify=verify, seed=s)
        counts[tuple(out)] += 1
    exp = joint.reshape(-1) * N
    obs = np.array([counts[(a, b)] for a in range(V) for b in range(V)])
    # pool rare cells so the chi-square approximation holds
    keep = exp >= 5
    obs_k = np.append(obs[keep], obs[~keep].sum())
    exp_k = np.append(exp[keep], exp[~keep].sum())
    if exp_k[-1] == 0:
        obs_k, exp_k = obs_k[:-1], exp_k[:-1]
    assert _chisq_pvalue(obs_k, exp_k) > 1e-3


def test_rejection_accepts_more_than_matching(models):
    """On identical contexts, rejection sampling's acceptance (sum min(p,q))
    dominates token matching's (sum p*q) at every temperature > 0."""
    from quantdraft.analysis import acceptance_profile

    target, drafts = models
    mx.random.seed(0)
    seqs = mx.random.randint(0, 64, (8, 48))
    for name in ("rtn2", "rtn4", "unrelated"):
        prof = acceptance_profile(target, drafts[name], seqs, temperatures=(0.0, 0.7, 1.0, 1.5))
        for T in ("0.7", "1", "1.5"):
            assert prof[f"alpha_rejection@{T}"] >= prof[f"alpha_match@{T}"], (name, T, prof)
    # a perfect draft is always accepted under rejection sampling...
    perfect = acceptance_profile(target, target, seqs, temperatures=(1.0,))
    assert perfect["alpha_rejection@1"] == pytest.approx(1.0, abs=1e-5)
    # ...but not under matching, unless the distribution is one-hot
    assert perfect["alpha_match@1"] < 0.999


def test_sampling_reproducible_with_seed(models):
    target, drafts = models
    a, _ = speculative_generate(target, drafts["rtn2"], PROMPT, 30, 3, 0.8, seed=7)
    b, _ = speculative_generate(target, drafts["rtn2"], PROMPT, 30, 3, 0.8, seed=7)
    assert a == b


def test_bad_arguments(models):
    target, drafts = models
    with pytest.raises(ValueError):
        speculative_generate(target, drafts["rtn2"], PROMPT, 5, verify="nope")
    with pytest.raises(ValueError):
        speculative_generate(target, drafts["rtn2"], [], 5)


def test_nested_draft_greedy_identical():
    t8 = rtn_draft(tiny_llama(seed=0, logit_scale=8.0), 8)
    d = nested_draft(t8, 3)
    ref, _ = autoregressive_generate(t8, PROMPT, 50)
    out, st = speculative_generate(t8, d, PROMPT, 50, gamma=4)
    assert out == ref and st.accepted > 0
