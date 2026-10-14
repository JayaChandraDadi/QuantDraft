"""The real CLI path on a tiny local model (no downloads)."""

import json

from quantdraft.cli import main

from .conftest import tiny_llama, write_byte_model_dir


def test_cli_generate_profile_bench_plot(tmp_path, capsys):
    model_dir = write_byte_model_dir(tmp_path / "tiny", tiny_llama(vocab=256, seed=0, logit_scale=6.0))
    main(["generate", "--target", str(model_dir), "--draft", "rtn:4", "--prompt", "def f(x):",
          "--max-tokens", "12", "--gamma", "3"])
    out = capsys.readouterr().out
    assert '"generated": 12' in out

    main(["profile", "--target", str(model_dir), "--draft", "rtn:3", "--context", "32",
          "--max-k", "4", "--repeats", "2", "--json", str(tmp_path / "cost.json")])
    cost = json.loads((tmp_path / "cost.json").read_text())
    assert set(cost["t_target"]) == {"1", "2", "3", "4"} and cost["t_draft"] > 0

    res = tmp_path / "res.jsonl"
    main(["bench", "--target", str(model_dir), "--draft", "rtn:3", "rtn:4",
          "--gammas", "2,auto", "--temps", "0,1.0", "--n-prompts", "2", "--max-tokens", "10",
          "--results", str(res), "--tag", "t"])
    rows = [json.loads(l) for l in res.read_text().splitlines()]
    base = [r for r in rows if r["kind"] == "baseline"]
    spec = [r for r in rows if r["kind"] == "speculative"]
    assert len(base) == 2
    # 2 drafts x (T=0: 2 gammas + T=1: 2 rules x 2 gammas) = 12
    assert len(spec) == 12
    assert all(r["identical_to_target"] for r in spec if r["temperature"] == 0)
    assert all("predicted_speedup" in r for r in spec if r["gamma"] == "2")

    main(["plot", "--results", str(res), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "alpha_by_draft.png").exists()
    assert (tmp_path / "fig" / "rejection_vs_match.png").exists()
