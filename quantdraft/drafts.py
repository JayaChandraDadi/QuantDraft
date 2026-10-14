"""Draft models built from the target itself.

Three families, all sharing the target's tokenizer and training:

* **RTN** -- round-to-nearest copy of the target at ``k`` bits
  (``mx.quantize``; what ``mlx_lm.convert -q`` produces).
* **BitBudget** -- a salience-guided mixed-precision copy at a bits-per-weight
  budget (https://github.com/JayaChandraDadi/BitBudget). Built with
  ``bitbudget quantize`` and loaded here like any mlx-lm model directory.
* **Nested (bit-plane)** -- derived from an already-quantized ``B``-bit target
  by keeping only the top ``k`` bits of every stored integer:

      q_k = q_B >> (B - k),   scale_k = scale * 2^(B-k),
      bias_k = bias + scale * (2^(B-k) - 1) / 2

  (the half-step offset re-centres each coarse bin). Every draft weight is a
  truncation of the *same* integer the target stores, so with a bit-plane
  storage layout the draft would cost no extra memory at all. MLX kernels
  expect contiguous ``k``-bit packing, so here the truncated integers are
  re-packed into a separate (small) array; a fused bit-plane Metal kernel is
  the natural next step.
"""

from __future__ import annotations

import copy

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_reduce, tree_unflatten


# --------------------------------------------------------------------------- #
# MLX affine packing: element i of a row occupies bits [i*b, (i+1)*b) of a
# little-endian bit stream stored in uint32 words (verified for b in
# {2,3,4,5,6,8} by tests/test_drafts.py::test_pack_matches_mlx).
# --------------------------------------------------------------------------- #
def unpack(packed: mx.array, bits: int) -> np.ndarray:
    """``(rows, words)`` uint32 -> ``(rows, cols)`` uint8 integers."""
    p = np.array(packed, dtype=np.uint32)
    rows = p.shape[0]
    stream = np.unpackbits(p.view(np.uint8).reshape(rows, -1), axis=1, bitorder="little")
    cols = stream.shape[1] // bits
    stream = stream[:, : cols * bits].reshape(rows, cols, bits)
    weights = (1 << np.arange(bits)).astype(np.uint16)
    return (stream * weights).sum(axis=-1).astype(np.uint8)


def pack(q: np.ndarray, bits: int) -> mx.array:
    """``(rows, cols)`` integers in ``[0, 2^bits)`` -> MLX packed uint32."""
    q = np.asarray(q, dtype=np.uint8)
    rows, cols = q.shape
    if (cols * bits) % 32:
        raise ValueError("cols * bits must be a multiple of 32")
    bitsarr = ((q[..., None] >> np.arange(bits, dtype=np.uint8)) & 1).reshape(rows, -1)
    packed = np.packbits(bitsarr, axis=1, bitorder="little")
    return mx.array(packed.view(np.uint32).reshape(rows, -1))


def truncate_quantized(layer: nn.QuantizedLinear, bits: int) -> nn.QuantizedLinear:
    """Top-``bits`` bit-plane copy of an affine-quantized linear layer."""
    if getattr(layer, "mode", "affine") != "affine":
        raise ValueError("only affine quantization can be truncated")
    src = layer.bits
    if bits >= src:
        raise ValueError(f"draft bits ({bits}) must be < target bits ({src})")
    shift = src - bits
    q = unpack(layer.weight, src) >> shift
    scales = layer.scales.astype(mx.float32)
    biases = layer.biases.astype(mx.float32)
    new_scales = scales * (1 << shift)
    new_biases = biases + scales * ((1 << shift) - 1) / 2.0

    out_dim, in_dim = q.shape
    new = nn.QuantizedLinear(in_dim, out_dim, bias="bias" in layer,
                             group_size=layer.group_size, bits=bits)
    new.weight = pack(q, bits)
    new.scales = new_scales.astype(layer.scales.dtype)
    new.biases = new_biases.astype(layer.biases.dtype)
    if "bias" in layer:
        new.bias = layer.bias
    new.freeze()
    return new


def nested_draft(target: nn.Module, bits: int) -> nn.Module:
    """Draft whose quantized linear layers are the top ``bits`` bit-planes of
    the (already quantized) target's. Non-quantized parts (norms,
    embeddings unless quantized) are shared by reference, not copied."""
    draft = _clone_tree(target)
    replaced = 0
    updates = []
    for name, mod in draft.named_modules():
        if isinstance(mod, nn.QuantizedLinear) and mod.bits > bits:
            updates.append((name, truncate_quantized(mod, bits)))
            replaced += 1
    if replaced == 0:
        raise ValueError("target has no quantized linear layers with more than "
                         f"{bits} bits; quantize it first (e.g. 8-bit)")
    draft.update_modules(tree_unflatten(updates))
    return draft


def _clone_tree(model: nn.Module) -> nn.Module:
    """Copy the module tree without duplicating parameter arrays."""
    clone = copy.copy(model)
    for k, v in model.items():
        if isinstance(v, nn.Module):
            clone[k] = _clone_tree(v)
        elif isinstance(v, list):
            clone[k] = [_clone_tree(x) if isinstance(x, nn.Module) else x for x in v]
        else:
            clone[k] = v
    return clone


def rtn_draft(target: nn.Module, bits: int, group_size: int = 64,
              quantize_embeddings: bool = True) -> nn.Module:
    """Round-to-nearest ``bits`` copy of the target. If the target is itself
    quantized, its layers are dequantized first (so this is RTN of the
    quantized target)."""
    draft = _clone_tree(target)
    deq = []
    for name, m in draft.named_modules():
        if isinstance(m, nn.QuantizedLinear):
            w = mx.dequantize(m.weight, m.scales, m.biases, m.group_size, m.bits)
            lin = nn.Linear(w.shape[1], w.shape[0], bias="bias" in m)
            lin.weight = w
            if "bias" in m:
                lin.bias = m.bias
            deq.append((name, lin))
        elif isinstance(m, nn.QuantizedEmbedding):
            w = mx.dequantize(m.weight, m.scales, m.biases, m.group_size, m.bits)
            emb = nn.Embedding(w.shape[0], w.shape[1])
            emb.weight = w
            deq.append((name, emb))
    if deq:
        draft.update_modules(tree_unflatten(deq))
    nn.quantize(
        draft,
        group_size=group_size,
        bits=bits,
        class_predicate=lambda p, m: (
            hasattr(m, "to_quantized")
            and m.weight.shape[-1] % group_size == 0
            and (quantize_embeddings or not isinstance(m, nn.Embedding))
        ),
    )
    return draft


def load_draft(spec: str, target: nn.Module) -> nn.Module:
    """``nested:K`` | ``rtn:K`` | path or HF repo of an mlx-lm model."""
    if spec.startswith("nested:"):
        return nested_draft(target, int(spec.split(":")[1]))
    if spec.startswith("rtn:"):
        return rtn_draft(target, int(spec.split(":")[1]))
    from mlx_lm import load

    return load(spec)[0]


def weight_bytes(model: nn.Module) -> int:
    """Bytes streamed per decode step ~ total parameter bytes."""
    return tree_reduce(lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc,
                       model.parameters(), 0)


def shared_bytes(a: nn.Module, b: nn.Module) -> int:
    """Bytes of parameter arrays that two models share by identity."""
    ids = {id(v): v.nbytes for _, v in tree_flatten(a.parameters())}
    return sum(v.nbytes for _, v in tree_flatten(b.parameters()) if id(v) in ids)
