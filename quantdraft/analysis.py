"""Teacher-forced acceptance analysis.

Running the full engine gives acceptance numbers that depend on the sampled
path. For comparing drafts it is cleaner to score both models on the *same*
contexts: run the target and the draft once over fixed token sequences and
compute, at every position,

* ``alpha_rejection = sum_v min(p_v, q_v)`` -- acceptance probability under
  the paper's rejection rule (``1 - total variation distance``),
* ``alpha_match     = sum_v p_v * q_v``     -- acceptance probability under
  token matching (mlx-lm's rule),
* ``greedy_agree    = [argmax p == argmax q]`` -- acceptance at temperature 0.

These are exactly the per-token acceptance probabilities the engine sees,
averaged over a fixed workload, with no sampling noise; two forward passes
per sequence regardless of how many drafts or gammas are compared.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def acceptance_profile(target: nn.Module, draft: nn.Module, sequences, temperatures=(0.0, 1.0),
                       batch_size: int = 8, skip: int = 1) -> dict:
    """Mean per-position acceptance statistics over ``sequences``
    (``(N, L)`` token array). ``skip`` leading positions are ignored (they
    have almost no context)."""
    seq = mx.array(sequences)
    acc = {"greedy_agree": []}
    for T in temperatures:
        if T > 0:
            acc[f"alpha_rejection@{T:g}"] = []
            acc[f"alpha_match@{T:g}"] = []
    for i in range(0, seq.shape[0], batch_size):
        x = seq[i : i + batch_size, :-1]
        lp = target(x).astype(mx.float32)[:, skip:]
        lq = draft(x).astype(mx.float32)[:, skip:]
        acc["greedy_agree"].append(
            (mx.argmax(lp, -1) == mx.argmax(lq, -1)).astype(mx.float32).reshape(-1))
        for T in temperatures:
            if T == 0:
                continue
            p = mx.softmax(lp / T, axis=-1)
            q = mx.softmax(lq / T, axis=-1)
            acc[f"alpha_rejection@{T:g}"].append(mx.minimum(p, q).sum(-1).reshape(-1))
            acc[f"alpha_match@{T:g}"].append((p * q).sum(-1).reshape(-1))
        mx.eval(acc)
    return {k: float(np.mean(np.concatenate([np.array(v) for v in vs]))) for k, vs in acc.items()}
