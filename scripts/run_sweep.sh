#!/usr/bin/env bash
# Real-model experiment on Apple silicon.
#
#   ./scripts/run_sweep.sh                                   # Qwen2.5-1.5B-Instruct target
#   TARGET=Qwen/Qwen2.5-0.5B-Instruct ./scripts/run_sweep.sh
#
# Target: the model at 8 bits (the "precise" model we want to speed up).
# Drafts: nested bit-plane (2/3/4), RTN (3/4), BitBudget (2.75/3.0/3.5 bpw),
#         and a separate smaller model from the same family (classic setup).
# Every draft is benchmarked at several gammas, greedy and sampled, with both
# verification rules; results land in one JSONL that `quantdraft plot` reads.
set -euo pipefail

TARGET=${TARGET:-Qwen/Qwen2.5-1.5B-Instruct}
SMALL=${SMALL:-Qwen/Qwen2.5-0.5B-Instruct}
NAME=${NAME:-$(basename "$TARGET")}
OUT=${OUT:-runs/$NAME}
MAX_TOKENS=${MAX_TOKENS:-128}
GAMMAS=${GAMMAS:-1,2,3,4,6,8,heuristic,auto}
TEMPS=${TEMPS:-0,0.7}
mkdir -p "$OUT"
RES="$OUT/results.jsonl"

# 1. the 8-bit target ---------------------------------------------------------
[ -d "$OUT/target8" ] || mlx_lm.convert --hf-path "$TARGET" -q --q-bits 8 --mlx-path "$OUT/target8"

# 2. drafts that need building -------------------------------------------------
DRAFTS=("nested:2" "nested:3" "nested:4" "rtn:3" "rtn:4")
if command -v bitbudget >/dev/null; then
  for b in 2.75 3.0 3.5; do
    [ -d "$OUT/bb$b" ] || bitbudget quantize --model "$TARGET" --target-bpw "$b" \
      --curves "$OUT/bitbudget_curves.json" --out "$OUT/bb$b"
    DRAFTS+=("$OUT/bb$b")
  done
else
  echo "bitbudget not installed (pip install -e '.[bitbudget]'); skipping BitBudget drafts"
fi
[ -d "$OUT/small4" ] || mlx_lm.convert --hf-path "$SMALL" -q --q-bits 4 --mlx-path "$OUT/small4"
DRAFTS+=("$OUT/small4")

# 3. cost profile of each draft (t_target(k), t_draft) ---------------------------
for d in "${DRAFTS[@]}"; do
  tag=$(echo "$d" | tr '/:' '__')
  [ -f "$OUT/cost_$tag.json" ] || quantdraft profile --target "$OUT/target8" --draft "$d" \
    --context 512 --json "$OUT/cost_$tag.json"
done

# 4. the benchmark ------------------------------------------------------------------
quantdraft bench --target "$OUT/target8" --draft "${DRAFTS[@]}" \
  --gammas "$GAMMAS" --temps "$TEMPS" --max-tokens "$MAX_TOKENS" \
  --tag "$NAME" --results "$RES"

# 5. figures ----------------------------------------------------------------------
quantdraft plot --results "$RES" --out "$OUT/figures" --title "$NAME (8-bit target)"
echo "done: $RES and $OUT/figures/"
