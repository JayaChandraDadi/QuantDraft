import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from quantdraft.drafts import (
    nested_draft,
    pack,
    rtn_draft,
    shared_bytes,
    truncate_quantized,
    unpack,
    weight_bytes,
)

from .conftest import tiny_llama


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_pack_matches_mlx(bits):
    rng = np.random.default_rng(bits)
    q = rng.integers(0, 2**bits, (16, 128))
    packed = pack(q, bits)
    # MLX itself must decode our packing to the same integers
    d = mx.dequantize(packed, mx.ones((16, 2)), mx.zeros((16, 2)), group_size=64, bits=bits)
    assert np.array_equal(np.array(d), q)
    assert np.array_equal(unpack(packed, bits), q)
    # and our unpacking must read MLX's own packing
    w, s, b = mx.quantize(mx.array(rng.normal(size=(16, 128)).astype(np.float32)), 64, bits)
    qi = unpack(w, bits).astype(np.float32)
    s_full = np.repeat(np.array(s), 64, axis=1)
    b_full = np.repeat(np.array(b), 64, axis=1)
    ref = np.array(mx.dequantize(w, s, b, group_size=64, bits=bits))
    assert np.allclose(s_full * qi + b_full, ref, atol=1e-5)


def _quantized_linear(bits, seed=0, in_dim=256, out_dim=64):
    mx.random.seed(seed)
    lin = nn.Linear(in_dim, out_dim, bias=False)
    return nn.QuantizedLinear.from_linear(lin, group_size=64, bits=bits), lin


@pytest.mark.parametrize("src,dst", [(8, 4), (8, 3), (8, 2), (4, 2), (6, 3)])
def test_truncation_is_exact_bitplane_math(src, dst):
    ql, _ = _quantized_linear(src)
    tl = truncate_quantized(ql, dst)
    q_src = unpack(ql.weight, src).astype(np.float64)
    shift = src - dst
    q_dst = np.floor(q_src / 2**shift)
    s = np.repeat(np.array(ql.scales, dtype=np.float64), 64, axis=1)
    b = np.repeat(np.array(ql.biases, dtype=np.float64), 64, axis=1)
    expected = s * (q_dst * 2**shift + (2**shift - 1) / 2) + b
    got = np.array(mx.dequantize(tl.weight, tl.scales, tl.biases, 64, dst))
    assert np.allclose(got, expected, atol=1e-5)
    assert tl.bits == dst and tl.weight.dtype == mx.uint32


def test_truncation_error_comparable_to_rtn():
    ql, lin = _quantized_linear(8, seed=3)
    for k in (2, 3, 4):
        tl = truncate_quantized(ql, k)
        w_nested = mx.dequantize(tl.weight, tl.scales, tl.biases, 64, k)
        rtn = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=k)
        w_rtn = mx.dequantize(rtn.weight, rtn.scales, rtn.biases, 64, k)
        e_nested = mx.mean((w_nested - lin.weight) ** 2).item()
        e_rtn = mx.mean((w_rtn - lin.weight) ** 2).item()
        # same grid resolution; nested bins are fixed by the 8-bit range so it
        # can be a little worse, never wildly so
        assert e_nested < 1.6 * e_rtn


def test_truncation_rejects_bad_bits():
    ql, _ = _quantized_linear(4)
    with pytest.raises(ValueError):
        truncate_quantized(ql, 4)


def test_rtn_draft_leaves_target_untouched_and_shares_norms():
    t = tiny_llama()
    before = t(mx.array([[1, 2, 3]]))
    d = rtn_draft(t, 3)
    assert isinstance(t.model.layers[0].mlp.up_proj, nn.Linear)
    assert isinstance(d.model.layers[0].mlp.up_proj, nn.QuantizedLinear)
    assert mx.array_equal(before, t(mx.array([[1, 2, 3]])))
    assert shared_bytes(t, d) > 0  # norm weights are shared, not copied
    assert weight_bytes(d) < weight_bytes(t) / 4


def test_rtn_draft_from_quantized_target():
    t8 = rtn_draft(tiny_llama(), 8)
    d = rtn_draft(t8, 3)
    assert d.model.layers[0].self_attn.q_proj.bits == 3
    assert t8.model.layers[0].self_attn.q_proj.bits == 8


def test_nested_draft_end_to_end():
    t8 = rtn_draft(tiny_llama(), 8)
    d = nested_draft(t8, 3)
    for name, m in d.named_modules():
        if isinstance(m, nn.QuantizedLinear):
            assert m.bits == 3, name
    out = d(mx.array([[1, 2, 3]]))
    assert out.shape == (1, 3, 64) and mx.isfinite(out).all().item()
    assert t8.model.layers[0].mlp.down_proj.bits == 8  # target untouched


def test_nested_draft_needs_quantized_target():
    with pytest.raises(ValueError):
        nested_draft(tiny_llama(), 3)
