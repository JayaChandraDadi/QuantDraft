"""Command line interface.

    quantdraft generate --target runs/qwen-8bit --draft runs/qwen-bb3.0 --prompt "..." --gamma auto
    quantdraft profile  --target runs/qwen-8bit --draft nested:3 --json cost.json
    quantdraft bench    --target runs/qwen-8bit --draft runs/qwen-bb3.0 nested:3 rtn:3 \\
                        --gammas 1,2,4,6,auto --temps 0,0.7 --results results.jsonl
    quantdraft plot     --results results.jsonl --out figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _append(path, rec):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _load_pair(a):
    from mlx_lm import load

    from .drafts import load_draft

    target, tok = load(a.target)
    draft = load_draft(a.draft, target) if getattr(a, "draft", None) else None
    return target, tok, draft


def _eos(tok):
    ids = getattr(tok, "eos_token_ids", None)
    if ids:
        return set(ids)
    e = getattr(tok, "eos_token_id", None)
    return {e} if e is not None else set()


def cmd_generate(a):
    from .controllers import make_controller
    from .engine import autoregressive_generate, speculative_generate
    from .prompts import encode_prompt

    target, tok, draft = _load_pair(a)
    prompt = encode_prompt(tok, a.prompt)
    if draft is None:
        out, st = autoregressive_generate(target, prompt, a.max_tokens, a.temp, a.seed, _eos(tok))
    else:
        out, st = speculative_generate(target, draft, prompt, a.max_tokens,
                                       make_controller(a.gamma), a.temp, a.verify, a.seed, _eos(tok))
    print(tok.decode(out))
    print("\n" + json.dumps(st.summary(), indent=1))


def cmd_profile(a):
    from .bench import profile_costs
    from .drafts import weight_bytes
    from .prompts import all_prompts, encode_prompt

    target, tok, draft = _load_pair(a)
    ctx = encode_prompt(tok, all_prompts()[0][1])
    ctx = (ctx * (a.context // max(len(ctx), 1) + 1))[: a.context]
    cm = profile_costs(target, draft, ctx, a.max_k, a.repeats)
    rec = {"target": a.target, "draft": a.draft, "context": a.context, **cm.to_dict(),
           "target_bytes": weight_bytes(target),
           "draft_bytes": weight_bytes(draft) if draft is not None else None}
    if cm.t_draft is not None:
        rec["cost_ratio"] = cm.t_draft / cm.t_target[1]
        rec["bytes_ratio"] = rec["draft_bytes"] / rec["target_bytes"]
    print(json.dumps(rec, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(rec, indent=1))


def cmd_bench(a):
    from mlx_lm import load

    from .bench import RunConfig, profile_costs, run_baseline, run_speculative, warmup
    from .drafts import load_draft, weight_bytes
    from .prompts import all_prompts, encode_prompt

    target, tok = load(a.target)
    eos = _eos(tok)
    items = all_prompts()
    if a.categories:
        items = [x for x in items if x[0] in a.categories.split(",")]
    if a.n_prompts:
        items = items[: a.n_prompts]
    prompts = [encode_prompt(tok, p) for _, p in items]
    temps = [float(t) for t in a.temps.split(",")]
    verifies = a.verify.split(",")
    gammas = a.gammas.split(",")

    warmup(target, None, prompts[0])
    baselines = {}
    for T in temps:
        outs, summ = run_baseline(target, prompts, a.max_tokens, T, a.seed, eos)
        baselines[T] = (outs, summ)
        _append(a.results, {"tag": a.tag, "kind": "baseline", "target": a.target,
                            "temperature": T, **summ})
        print(f"baseline T={T}: {summ['decode_tps']:.1f} tok/s")

    for spec in a.draft:
        draft = load_draft(spec, target)
        warmup(target, draft, prompts[0])
        cost = profile_costs(target, draft, prompts[0][-64:] * 4, max_k=13) if a.profile else None
        tb, db = weight_bytes(target), weight_bytes(draft)
        for T in temps:
            for v in verifies:
                if T == 0 and v != verifies[0]:
                    continue  # identical rules at T = 0
                for g in gammas:
                    cfg = RunConfig(gamma=g, temperature=T, verify=v)
                    outs, summ = run_speculative(target, draft, prompts, a.max_tokens, cfg,
                                                 a.seed, eos, cost, a.max_gamma)
                    base = baselines[T][1]
                    rec = {
                        "tag": a.tag, "kind": "speculative", "target": a.target, "draft": spec,
                        "gamma": g, "temperature": T, "verify": v,
                        "target_bytes": tb, "draft_bytes": db, "bytes_ratio": db / tb,
                        **summ,
                        "baseline_tps": base["decode_tps"],
                        "speedup": summ["decode_tps"] / base["decode_tps"] if base["decode_tps"] else None,
                    }
                    if T == 0:
                        rec["identical_to_target"] = outs == baselines[T][0]
                    if cost is not None:
                        rec["cost"] = cost.to_dict()
                        rec["cost_ratio"] = cost.t_draft / cost.t_target[1]
                        if g.isdigit() and int(g) > 0:
                            rec["predicted_speedup"] = cost.speedup(summ["alpha"], int(g))
                    _append(a.results, rec)
                    print(f"{spec:28s} {cfg.name:28s} alpha={summ['alpha']:.3f} "
                          f"tok/round={summ['tokens_per_round']:.2f} "
                          f"speedup={rec['speedup'] or 0:.2f}x"
                          + ("" if T else f" identical={rec['identical_to_target']}"))
        del draft


def cmd_plot(a):
    from .plotting import plot_all

    for p in plot_all(a.results, a.out, a.title):
        print(p)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="quantdraft")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, draft_nargs=None):
        p.add_argument("--target", required=True, help="mlx-lm model dir or HF repo")
        if draft_nargs:
            p.add_argument("--draft", nargs=draft_nargs, required=True,
                           help="draft model dir/repo, or nested:K / rtn:K")
        else:
            p.add_argument("--draft", help="draft model dir/repo, or nested:K / rtn:K")

    g = sub.add_parser("generate")
    common(g)
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-tokens", type=int, default=256)
    g.add_argument("--gamma", default="auto", help="int, 'heuristic' or 'auto'")
    g.add_argument("--temp", type=float, default=0.0)
    g.add_argument("--verify", choices=["rejection", "match"], default="rejection")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(fn=cmd_generate)

    p = sub.add_parser("profile", help="measure t_target(k) and t_draft")
    common(p)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--max-k", type=int, default=13)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--json")
    p.set_defaults(fn=cmd_profile)

    b = sub.add_parser("bench")
    common(b, draft_nargs="+")
    b.add_argument("--gammas", default="1,2,3,4,6,8,heuristic,auto")
    b.add_argument("--temps", default="0,0.7")
    b.add_argument("--verify", default="rejection,match")
    b.add_argument("--max-tokens", type=int, default=128)
    b.add_argument("--max-gamma", type=int, default=12)
    b.add_argument("--n-prompts", type=int, default=0)
    b.add_argument("--categories")
    b.add_argument("--no-profile", dest="profile", action="store_false")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--tag", default="run")
    b.add_argument("--results")
    b.set_defaults(fn=cmd_bench)

    pl = sub.add_parser("plot")
    pl.add_argument("--results", required=True)
    pl.add_argument("--out", default="figures")
    pl.add_argument("--title")
    pl.set_defaults(fn=cmd_plot)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
