"""Toy-scale validation of QuantDraft (runs anywhere MLX runs, no downloads).

Target: the 3.0M-parameter byte-level Llama from BitBudget's toy run,
quantized to 8 bits (the "precise" model we want to speed up).

Drafts, all sharing the target's tokenizer and training:
    nested-k     top k bit-planes of the 8-bit target's own integers
    rtn-k        round-to-nearest k-bit copy of the bf16 weights
    bitbudget-b  BitBudget mixed precision at b bits/weight (AWQ + Fisher + knapsack)
    small        a separately trained 0.18M-parameter byte model (classic draft)

Measured here (hardware-independent):
  * teacher-forced acceptance on held-out text: greedy agreement,
    alpha_rejection = sum min(p, q) and alpha_match = sum p*q at T = 0.7, 1.0
  * real speculative decoding runs: greedy output identical to the target,
    tokens per target forward for several gamma

Projected (clearly labelled): speedup on memory-bound hardware from the
measured acceptance and the drafts' weight bytes. Wall-clock speedups must be
measured on Apple silicon (scripts/run_sweep.sh); this machine's MLX CPU
backend has no BLAS and says nothing about Apple GPU timing.

Quality is evaluated with dequantized fp32 weights (numerically the same
weights; MLX's quantized kernels are ~13x slower than dense on Linux CPU).

    python examples/toy_validation.py \\
        --weights ../BitBudget/results/toy/toy_model.npz \\
        --curves ../BitBudget/results/toy/curves_fisher.json \\
        --small-draft results/toy/small_draft/toy_model.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm.models import llama  # noqa: E402
from mlx_lm.utils import dequantize_model  # noqa: E402

from quantdraft.analysis import acceptance_profile  # noqa: E402
from quantdraft.costmodel import expected_tokens  # noqa: E402
from quantdraft.drafts import _clone_tree, nested_draft, rtn_draft, shared_bytes, weight_bytes  # noqa: E402
from quantdraft.engine import autoregressive_generate, speculative_generate  # noqa: E402


def load_corpus(root: Path, max_bytes: int = 8_000_000):
    """Same deterministic file-level split as BitBudget's toy run."""
    train, test = bytearray(), bytearray()
    for p in sorted(p for p in root.rglob("*.py") if "test" not in p.parts
                    and "site-packages" not in p.parts):
        held = int(hashlib.md5(str(p.relative_to(root)).encode()).hexdigest(), 16) % 10 == 0
        (test if held else train).extend(p.read_bytes() + b"\n")
        if len(train) > max_bytes:
            break
    as_np = lambda b: np.frombuffer(bytes(b), dtype=np.uint8).astype(np.int32)
    return as_np(train[:max_bytes]), as_np(test)


def make_model(hidden, layers, inter):
    return llama.Model(llama.ModelArgs(
        model_type="llama", hidden_size=hidden, num_hidden_layers=layers,
        intermediate_size=inter, num_attention_heads=hidden // 64,
        num_key_value_heads=max(1, hidden // 128), rms_norm_eps=1e-5, vocab_size=256,
        tie_word_embeddings=True))


def load_npz(model, path, dtype=mx.bfloat16):
    w = np.load(path)
    model.load_weights([(k, mx.array(w[k])) for k in w.files])
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    return model


def dense_fp32(model):
    """Dequantized fp32 copy (same weights) for fast CPU evaluation."""
    m = _clone_tree(model)
    dequantize_model(m)
    m.set_dtype(mx.float32)
    mx.eval(m.parameters())
    return m


def best_projection(alpha, ratio, max_gamma=12):
    best = (1.0, 0)
    for g in range(1, max_gamma + 1):
        s = expected_tokens(alpha, g) / (g * ratio + 1.0)
        if s > best[0]:
            best = (s, g)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--curves", help="BitBudget curves_fisher.json (enables BitBudget drafts)")
    ap.add_argument("--small-draft", help="small separately trained draft (.npz)")
    ap.add_argument("--corpus", default="/usr/lib/python3.13")
    ap.add_argument("--n-eval", type=int, default=48)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--gen-tokens", type=int, default=96)
    ap.add_argument("--out", default="results/toy")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    train, test = load_corpus(Path(args.corpus))
    n = test.size // args.seq_len
    evals = mx.array(test[: n * args.seq_len].reshape(n, args.seq_len)[: args.n_eval])

    base = load_npz(make_model(256, 4, 704), args.weights)
    target8 = rtn_draft(base, 8)  # the precise model being accelerated
    target_bytes = weight_bytes(target8)
    print(f"target: 8-bit, {target_bytes/1e6:.2f} MB of weights", flush=True)

    drafts = {}
    for k in (2, 3, 4):
        drafts[f"nested-{k}"] = ("nested", nested_draft(target8, k))
    for k in (2, 3, 4):
        drafts[f"rtn-{k}"] = ("rtn", rtn_draft(base, k))
    if args.curves:
        from bitbudget import Config, Curves, allocate, apply_plan, quantize_in_memory

        curves = Curves.load(args.curves)
        calib = mx.array(train[-2_000_000:][: 8 * 256].reshape(8, 256))
        for b in (2.75, 3.0, 3.5):
            m = load_npz(make_model(256, 4, 704), args.weights)
            plan = allocate(curves, b)["plan"]
            apply_plan(m, calib, plan, Config(bits=tuple(curves.bits), n_grid=10, verbose=False))
            quantize_in_memory(m, plan, embed_bits=4)
            drafts[f"bitbudget-{b}"] = ("bitbudget", m)
    if args.small_draft:
        drafts["small-0.18M"] = ("small", load_npz(make_model(128, 1, 256), args.small_draft))

    target_eval = dense_fp32(target8)
    rows = []
    print("\nteacher-forced acceptance on held-out code", flush=True)
    for name, (family, d) in drafts.items():
        t0 = time.time()
        prof = acceptance_profile(target_eval, dense_fp32(d), evals, temperatures=(0.0, 0.7, 1.0))
        db = weight_bytes(d)
        extra = db - shared_bytes(target8, d)
        ratio = db / target_bytes
        sp_g, g_g = best_projection(prof["greedy_agree"], ratio)
        sp_s, g_s = best_projection(prof["alpha_rejection@0.7"], ratio)
        row = {"draft": name, "family": family, "draft_bytes": db, "extra_bytes": extra,
               "bytes_ratio": ratio, "extra_ratio": extra / target_bytes, **prof,
               "projected_speedup_greedy": sp_g, "projected_gamma_greedy": g_g,
               "projected_speedup_T0.7": sp_s, "projected_gamma_T0.7": g_s}
        rows.append(row)
        print(f"  {name:16s} bytes {ratio:5.3f}x (extra {extra/target_bytes:5.3f}x)  "
              f"greedy {prof['greedy_agree']:.3f}  rej@0.7 {prof['alpha_rejection@0.7']:.3f}  "
              f"match@0.7 {prof['alpha_match@0.7']:.3f}  proj {sp_g:.2f}x@g{g_g} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- real speculative decoding runs ----------------------------------
    print("\nspeculative decoding runs (greedy)", flush=True)
    prompts = [list(np.asarray(test[i * 997 : i * 997 + 48])) for i in range(args.n_prompts)]
    prompts = [[int(t) for t in p] for p in prompts]
    refs = [autoregressive_generate(target_eval, p, args.gen_tokens)[0] for p in prompts]
    runs = []
    pick = [n for n in drafts if n in ("nested-3", "rtn-3", "bitbudget-3.0", "small-0.18M",
                                       "nested-4", "nested-2")]
    for name in pick:
        d = dense_fp32(drafts[name][1])
        alpha = next(r["greedy_agree"] for r in rows if r["draft"] == name)
        for g in (1, 2, 3, 4, 6, 8):
            tok = rnd = 0
            identical = True
            for p, ref in zip(prompts, refs):
                o, st = speculative_generate(target_eval, d, p, args.gen_tokens, gamma=g)
                identical &= o == ref
                tok += st.generated
                rnd += st.rounds
            runs.append({"draft": name, "gamma": g, "tokens_per_round": tok / rnd,
                         "iid_prediction": expected_tokens(alpha, g), "identical": identical})
            print(f"  {name:16s} gamma={g}  tokens/target-forward {tok/rnd:.2f}  "
                  f"(i.i.d. model {expected_tokens(alpha, g):.2f})  identical={identical}",
                  flush=True)

    # sampled run sanity: rejection vs match on the same draft
    sampled = []
    for name in ("nested-3", "rtn-3"):
        if name not in drafts:
            continue
        d = dense_fp32(drafts[name][1])
        for v in ("rejection", "match"):
            tok = rnd = 0
            for s, p in enumerate(prompts):
                o, st = speculative_generate(target_eval, d, p, args.gen_tokens, gamma=4,
                                             temperature=0.7, verify=v, seed=s)
                tok += st.generated
                rnd += st.rounds
            sampled.append({"draft": name, "verify": v, "temperature": 0.7,
                            "tokens_per_round": tok / rnd})
            print(f"  {name:16s} T=0.7 {v:9s} tokens/target-forward {tok/rnd:.2f}", flush=True)

    (out / "acceptance.json").write_text(json.dumps(
        {"target_bytes": target_bytes, "drafts": rows, "runs": runs, "sampled": sampled,
         "seconds": time.time() - t_start}, indent=1))
    make_figures(rows, runs, sampled, out)
    print(f"\nwrote {out}/acceptance.json and figures in {time.time()-t_start:.0f}s")


FAMILY_STYLE = {"nested": ("#dc2626", "o"), "rtn": ("#6b7280", "s"),
                "bitbudget": ("#2563eb", "D"), "small": ("#059669", "^")}


def make_figures(rows, runs, sampled, out):
    # (1) acceptance vs extra memory
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    for ax, key, xkey, xl in (
        (axes[0], "greedy_agree", "bytes_ratio", "draft weight bytes / 8-bit target"),
        (axes[1], "greedy_agree", "extra_ratio", "extra memory for the draft / 8-bit target"),
    ):
        for fam, (c, mk) in FAMILY_STYLE.items():
            rs = [r for r in rows if r["family"] == fam]
            if not rs:
                continue
            rs.sort(key=lambda r: r[xkey])
            ax.plot([r[xkey] for r in rs], [r[key] for r in rs], marker=mk, color=c,
                    label=fam, lw=1.2)
            for r in rs:
                ax.annotate(r["draft"].split("-", 1)[-1], (r[xkey], r[key]), fontsize=6,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(xl)
        ax.set_ylabel("greedy acceptance (held-out)")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Toy target (3.0M params, 8-bit): draft acceptance vs size", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "toy_acceptance_vs_bytes.png")
    plt.close(fig)

    # (2) rejection vs matching acceptance
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    names = [r["draft"] for r in rows]
    x = np.arange(len(names))
    w = 0.2
    for j, (k, c, lab) in enumerate((
            ("alpha_rejection@0.7", "#dc2626", "rejection, T=0.7"),
            ("alpha_match@0.7", "#fca5a5", "matching, T=0.7"),
            ("alpha_rejection@1", "#1d4ed8", "rejection, T=1.0"),
            ("alpha_match@1", "#93c5fd", "matching, T=1.0"))):
        ax.bar(x + (j - 1.5) * w, [r[k] for r in rows], width=w, color=c, label=lab)
    ax.set_xticks(x, names, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("per-token acceptance probability")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Same drafts, same contexts: rejection sampling vs token matching", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "toy_rejection_vs_match.png")
    plt.close(fig)

    # (3) tokens per target forward vs gamma (measured) with i.i.d. prediction
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
    markers = {"2": "v", "3": "o", "4": "^"}
    for i, name in enumerate(dict.fromkeys(r["draft"] for r in runs)):
        rs = [r for r in runs if r["draft"] == name]
        fam = next(r["family"] for r in rows if r["draft"] == name)
        c = FAMILY_STYLE[fam][0]
        mk = markers.get(name[-1], "o") if fam == "nested" else FAMILY_STYLE[fam][1]
        ls = "-" if name.endswith("3") or name.endswith("3.0") or fam == "small" else "--"
        ax.plot([r["gamma"] for r in rs], [r["tokens_per_round"] for r in rs], marker=mk,
                color=c, ls=ls, label=name)
        ax.plot([r["gamma"] for r in rs], [r["iid_prediction"] for r in rs], color=c,
                ls=":", lw=0.8)
    ax.set_xlabel("draft length γ")
    ax.set_ylabel("tokens per target forward (measured)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Greedy speculative decoding (dotted: i.i.d. model)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "toy_tokens_per_round.png")
    plt.close(fig)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--replot":
        d = json.loads(Path(sys.argv[2], "acceptance.json").read_text())
        make_figures(d["drafts"], d["runs"], d["sampled"], Path(sys.argv[2]))
    else:
        main()
