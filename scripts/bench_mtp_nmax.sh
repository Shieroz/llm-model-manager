#!/usr/bin/env bash
#
# bench_mtp_nmax.sh — sweep llama.cpp's --spec-draft-n-max for an MTP-capable GGUF
# and report generation throughput + draft-token acceptance, to find the optimal value.
#
# How it works
#   For each n-max value it (re)deploys a single throwaway config through the
#   model-manager API (so llama-swap keeps only ONE model on the GPU at a time —
#   safe on a single 24 GB card), waits for the server, sends one warm-up request
#   and one measured request, then parses the server-reported `timings`.
#   n-max = 0 means "MTP disabled" (baseline, no --spec-type).
#   The throwaway config is removed on exit.
#
# Reuse for other models: override the env vars below, e.g.
#   REPO=unsloth/Some-MTP-GGUF QUANT=UD-Q4_K_XL MMPROJ= NMAX_LIST="0 1 2 3 4" \
#     ./scripts/bench_mtp_nmax.sh
#
# Requirements: the llm-model-manager (:8000) and llama-swap (:8080) stack running,
# python3, curl. The model must be an MTP-prepared GGUF for n-max > 0 to do anything.
#
set -euo pipefail

MANAGER_URL="${MANAGER_URL:-http://localhost:8000}"
SWAP_URL="${SWAP_URL:-http://localhost:8080}"

REPO="${REPO:-unsloth/Qwen3.6-27B-MTP-GGUF}"
QUANT="${QUANT:-UD-Q4_K_XL}"
MMPROJ="${MMPROJ:-F16}"                 # set MMPROJ= (empty) for text-only models
REVISION="${REVISION:-latest}"
BENCH_NAME="${BENCH_NAME:-mtp-bench}"   # throwaway config / symlink name
NMAX_LIST="${NMAX_LIST:-0 1 2 3 4 5}"   # 0 = baseline (MTP off)
MAX_TOKENS="${MAX_TOKENS:-256}"
PROMPT="${PROMPT:-Write a detailed step-by-step explanation of how the quicksort algorithm works, including its time complexity analysis.}"

# Base llama-server params (JSON object body, WITHOUT the spec-* flags — those are
# added per iteration). temp:0 keeps generation deterministic for a fair compare.
BASE_PARAMS="${BASE_PARAMS:-{\"ngl\":99,\"c\":32768,\"fa\":\"on\",\"cache-type-k\":\"q8_0\",\"cache-type-v\":\"q8_0\",\"t\":8,\"n\":-1,\"temp\":0,\"top-p\":0.95,\"top-k\":20,\"min-p\":0}}"

MODEL_ID="${BENCH_NAME}-${QUANT}"

cleanup() {
  echo ">> Removing throwaway config '${BENCH_NAME}'..."
  curl -s -X DELETE "${MANAGER_URL}/api/configs/${BENCH_NAME}" >/dev/null || true
}
trap cleanup EXIT

# JSON-encode the prompt safely (handles quotes/newlines).
PROMPT_JSON="$(printf '%s' "$PROMPT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"

# Build the parameters JSON for a given n-max ($1). n-max 0 => no MTP flags.
build_params() {
  local n="$1"
  if [ "$n" -eq 0 ]; then
    printf '%s' "$BASE_PARAMS"
  else
    python3 - "$BASE_PARAMS" "$n" <<'PY'
import json, sys
base = json.loads(sys.argv[1]); n = int(sys.argv[2])
base["spec-type"] = "draft-mtp"
base["spec-draft-n-max"] = n
print(json.dumps(base))
PY
  fi
}

# Deploy/redeploy the throwaway config with the given parameters JSON ($1).
deploy() {
  local params="$1"
  # `parameters` must be a JSON *string*, so re-encode it.
  local params_str; params_str="$(printf '%s' "$params" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
  curl -s -X POST "${MANAGER_URL}/api/setup" -H 'Content-Type: application/json' -d "{
    \"hf_repo\": \"${REPO}\", \"quant\": \"${QUANT}\", \"mmproj\": \"${MMPROJ}\",
    \"symlink_name\": \"${BENCH_NAME}\", \"parameters\": ${params_str}, \"revision\": \"${REVISION}\"
  }" >/dev/null
}

# Send one chat-completion; retries while llama-swap restarts / loads the model.
# Echoes the raw JSON (with `timings`) on success.
call_model() {
  local out
  for _ in $(seq 1 120); do
    out="$(curl -s --max-time 600 "${SWAP_URL}/v1/chat/completions" -H 'Content-Type: application/json' -d "{
      \"model\": \"${MODEL_ID}\",
      \"messages\": [{\"role\": \"user\", \"content\": ${PROMPT_JSON}}],
      \"max_tokens\": ${MAX_TOKENS}, \"temperature\": 0, \"seed\": 42, \"stream\": false
    }")" || true
    if printf '%s' "$out" | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("timings") else 1)' 2>/dev/null; then
      printf '%s' "$out"; return 0
    fi
    sleep 2
  done
  return 1
}

printf '\n=== MTP n-max sweep — %s (%s) ===\n' "$REPO" "$QUANT"
printf 'prompt=%d tokens=%s\n\n' "$(printf '%s' "$PROMPT" | wc -w)" "$MAX_TOKENS"
printf '%-7s %-12s %-10s %-10s %-10s\n' "n-max" "tok/s" "draft_n" "accepted" "accept%"
printf '%s\n' "-------------------------------------------------------"

BASELINE=""
RESULTS=""
for n in $NMAX_LIST; do
  deploy "$(build_params "$n")"
  call_model >/dev/null || { printf '%-7s %s\n' "$n" "FAILED to load"; continue; }   # warm-up
  json="$(call_model)" || { printf '%-7s %s\n' "$n" "FAILED to measure"; continue; }  # measured

  line="$(printf '%s' "$json" | python3 -c '
import json, sys
t = json.load(sys.stdin)["timings"]
tps = t.get("predicted_per_second", 0.0)
dn  = t.get("draft_n")
da  = t.get("draft_n_accepted")
acc = (da / dn * 100) if dn else 0.0
dn_s = "-" if dn is None else str(dn)
da_s = "-" if da is None else str(da)
print(f"{tps:.2f}|{dn_s}|{da_s}|{acc:.1f}")
')"
  tps="${line%%|*}"; rest="${line#*|}"; dn="${rest%%|*}"; rest="${rest#*|}"; da="${rest%%|*}"; acc="${rest##*|}"
  label="$n"; [ "$n" -eq 0 ] && label="0(off)"
  printf '%-7s %-12s %-10s %-10s %-10s\n' "$label" "$tps" "$dn" "$da" "$acc"
  [ "$n" -eq 0 ] && BASELINE="$tps"
  RESULTS="${RESULTS}${n} ${tps}\n"
done

echo
if [ -n "$BASELINE" ] && [ "$(printf '%s' "$BASELINE" | cut -d. -f1)" -gt 0 ] 2>/dev/null; then
  echo "Speedup vs baseline (n-max=0):"
  printf "$RESULTS" | while read -r n tps; do
    [ "$n" -eq 0 ] && continue
    spd="$(python3 -c "print(f'{${tps}/${BASELINE}:.2f}x  (+{(${tps}/${BASELINE}-1)*100:.0f}%)')")"
    printf '  n-max=%-3s %s\n' "$n" "$spd"
  done
  best="$(printf "$RESULTS" | sort -k2 -gr | head -1)"
  echo
  echo "Best: n-max=$(printf '%s' "$best" | awk '{print $1}') at $(printf '%s' "$best" | awk '{print $2}') tok/s"
fi
