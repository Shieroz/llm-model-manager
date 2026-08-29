# LLM Model Manager

Manage HuggingFace GGUF models with a single-panel UI. Downloads, pins revisions, and auto-generates llama-swap configs.

## Quick Start

```bash
export MODELS=/path/to/models HF_TOKEN=hf_...
./up.sh
```

Open `http://localhost:8000` → deploy a model → it appears at `http://localhost:8080` (llama-swap API).

## Backends

```bash
LLAMA_BACKEND=vulkan ./up.sh   # AMD/Intel Vulkan
LLAMA_BACKEND=sycl ./up.sh     # Intel SYCL
LLAMA_BACKEND=openvino ./up.sh # Intel OpenVINO
```

## How It Works

Two containers share a volume at `/models`:

| Container | Port | Role |
|---|---|---|
| `llm-model-manager` | 8000 | UI + API — downloads GGUF from HF, writes `config.yaml` |
| `llama-swap` | 8080 | llama-server proxy — reads `config.yaml`, serves models |

State is persisted in `/models/served/state.json`. Model cache lives at `/models/.cache` (HF cache).

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MODELS` | *(required)* | Host path mounted as `/models` |
| `HF_TOKEN` | *(required)* | HuggingFace token for gated models |
| `LOG_LEVEL` | `INFO` | Python log level |
| `LLAMA_BACKEND` | `cuda` | GPU backend: `cuda`, `vulkan`, `sycl`, `openvino` |
| `LLAMA_DOCKERFILE` | `Dockerfile.llama.cuda` | llama-swap Dockerfile to use |

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Frontend UI |
| `GET` | `/api/models` | List cached models & revisions |
| `GET` | `/api/quants?repo=...` | Available quants, mmproj projectors & MTP draft heads for a HF repo |
| `GET` | `/api/commits?repo=...` | Commit history with pin status |
| `POST` | `/api/setup` | Deploy a model (downloads if needed) |
| `DELETE` | `/api/configs/{name}` | Remove a config (auto-prunes cache) |
| `POST` | `/api/revisions/delete` | Delete a cached revision |
| `POST` | `/api/rpc_mode` | Toggle RPC mode for combining GPU power |
| `WS` | `/ws` | Real-time state updates |

## RPC Mode

Enable RPC mode to combine GPU power across multiple machines. Only models with `"rpc": true` in their parameters are synced when RPC mode is active. Toggle via the RPC Mode switch in the UI or:

```bash
curl -X POST http://localhost:8000/api/rpc_mode -H 'Content-Type: application/json' -d '{"enabled": true}'
```

## MTP (Multi-Token Prediction)

llama-server is built with MTP speculative decoding. The llama.cpp version is pinned
via `LLAMA_CPP_REF` (single source of truth in `docker-compose.yml`, overridable in
`.env`) — bump it to rebuild against a newer commit; changing it busts the cached
git-clone layer.

MTP is **model-specific** — it only works with MTP-prepared GGUFs, not arbitrary models.
The deploy form handles three distribution patterns:

1. **Grafted GGUFs** (MTP layers baked into the quant file, e.g. `unsloth/Qwen3.6-27B-MTP-GGUF`):
   detected automatically from the `mtp` token in the repo name — the form injects
   `"spec-type": "draft-mtp"` (+ a max-draft-tokens field) into the parameters JSON for you.
   Remove it there to disable. Verified ~1.7× speedup on Qwen3.6-27B at `spec-draft-n-max: 4`
   (see `scripts/bench_mtp_nmax.sh`).
2. **Separate head, same repo** (purpose-built draft module shipped alongside the main quant,
   e.g. gemma-4's `mtp-gemma-4-31B-it.gguf` in `unsloth/gemma-4-31B-it-GGUF`): the form shows a
   **draft-head dropdown** (populated from `/api/quants`). Pick a head and the manager downloads
   it, symlinks it to `<name>-mtp-head.gguf`, and passes `--spec-type draft-mtp --model-draft …`
   to llama-server automatically — no manual params needed.
3. **Separate head, different repo**: not auto-managed. Deploy the head repo as its own config,
   then copy its served symlink path (shown on each **Disk Storage** card) into the main model's
   `"model-draft"` parameter by hand.

⚠️ Not every `*-MTP.gguf` works as a runtime draft — some "head-only" repos are **graft sources**
for a `convert.py` step (incomplete hparams, declare the full base arch) and must be grafted into
the base GGUF first. The grafted-name heuristic only gates auto-injection of the spec flags; the
head dropdown is driven by actual file detection.

⚠️ **The name heuristic misses grafted repos that don't say `mtp`.** `unsloth/Qwen3.8-27B-GGUF`
ships the MTP layer *inside* the main quant (`blk.64.nextn.*`) but has no `mtp` token in the repo
name, so the form does not auto-inject and instead offers the `MTP/mtp-*.gguf` head from the
dropdown. Picking it wires up a redundant `--model-draft` that wastes ~1.3 GB of VRAM — the head
is already in the file. Before selecting a head, check whether the quant already contains one:

```bash
head -c 120M model.gguf | strings -n 5 | grep -c 'nextn'   # >0 → grafted, leave the dropdown empty
```

Pattern #1 applies whenever those tensors are present, regardless of the repo name.

> **Pin note:** `LLAMA_CPP_REF` is `6fe7498` — it carries MTP (PR #22673) and the
> `ngram-mod` drafter the Qwen3.8 configs rely on. Bumping it busts the cached
> git-clone layer and rebuilds llama-server.

## Performance Tuning

### ⚠️ Never set `-fitc` without `-ngl`

`-fitc` / `--fit-ctx` is a **minimum context floor**, not a context setting. With `-ngl` left at
its `auto` default, `--fit` honours that floor by **evicting model layers to CPU** instead of
shrinking context — silently, with no error. Measured cost on Qwen3.8-27B (RTX 3090): 3.7 tok/s
with ~7 GB of weights stranded in host RAM, versus 66 tok/s fully offloaded. Prompt eval still
looks fast (~600 tok/s), so the only obvious symptom is collapsed token generation.

**Always pin `-ngl` explicitly and set `-c` yourself.** With `-ngl` set, `--fit` refuses to
degrade and says so:

```
W common_fit_params: failed to fit params to free device memory:
  n_gpu_layers already set by user to 99, abort
```

That is the desired failure mode — a loud abort you can tune against, rather than a silent 17×
slowdown. Prefer `"ngl": 99` + an explicit `"c"` in every config's parameters JSON.

### Reference configs: Qwen3.8-27B on a single 24 GB card

Two configs off the same cached weights (`unsloth/Qwen3.8-27B-GGUF` @ `UD-Q4_K_M`, no separate
draft head — the MTP layer is already in the quant):

**`Qwen3.8-27B-NoVision`** — max context, 262144:

```json
{"ngl": 99, "c": 262144, "np": 1, "fa": "on",
 "cache-type-k": "q4_0", "cache-type-v": "q4_0", "cram": 32768,
 "spec-type": "draft-mtp,ngram-mod", "spec-draft-n-max": 2,
 "t": 8, "n": -1, "temp": 0.6, "top-p": 0.95, "top-k": 40, "min-p": 0}
```

23,120 MiB VRAM (1,456 MiB headroom), **1,048 tok/s PP**, **62-66 tok/s TG**.

**`Qwen3.8-27B-Vision`** — same but `"mmproj": "F16"` and `"c": 196608`. 22,464 MiB idle,
22,516 MiB peak while encoding an 896x896 image, so vision costs ~0.9 GB resident and only ~50 MiB
transient. Context drops to 196k to keep the headroom.

Why it fits: Qwen3.8-27B is a **hybrid** — of its 65 layers only 17 carry a growing KV cache
(`blk.*.attn_*`); the other 48 are constant-state SSM layers (`blk.*.ssm_*`). Long context is
far cheaper here than on a dense 27B, so there is no need to drop to a smaller quant.

Measured on this hardware (RTX 3090, `LLAMA_CPP_REF=6fe7498`):

| Change | Effect |
|---|---|
| `-fitc` floor -> explicit `-ngl 99` | **3.7 -> 66 tok/s** (the whole ballgame) |
| Drop redundant `--model-draft` | -1.3 GB VRAM, MTP still active |
| Drop `--mmproj` | -0.9 GB VRAM |
| `--cache-type-* q8_0` -> `q4_0` | 131k -> 262k context |
| `-np` auto (4 slots) -> `-np 1` | MTP gain shrinks as slots rise; 1 is right for single-user |
| `--spec-draft-n-max` 3 -> 2 | 50.6 -> 56.9 tok/s (24 GB cards peak at 2) |
| `+ ngram-mod` alongside `draft-mtp` | +7-11 % on repetitive/agentic work, neutral on prose |
| `-cram 32768` | long prompt re-entry 43 s -> 0.9 s (see below) |

### Tuning dead ends (measured, do not re-try)

The reference config sits at a local optimum. Everything below was tested against it and lost:

| Variant | PP | TG | Verdict |
|---|---|---|---|
| baseline (`-ub 512`, n-max 2) | 1048 | 61.5-63.8 | **best** |
| `-ub 1024` | 1084 | 62.3 | +3 % PP but headroom -> 794 MiB — violates the headroom rule |
| `-ub 2048` | — | — | OOM at load: `failed to allocate compute pp buffers` |
| `--spec-draft-n-max 3` (with ngram) | 1042 | 57.5 | worse |
| `--spec-ngram-mod-n-min 32 --n-max 96` | 1030 | 60.3 | worse |
| `--spec-ngram-mod-n-match 12` | — | 67.1 vs 69.9 | worse; upstream warns below 16 |
| `+ ngram-map-k4v` (third drafter) | 1049 | 61.3 | no change |
| `--spec-draft-p-min 0.6` | 1049 | 56.2 | clearly worse |
| `--backend-sampling` (`-bs`) | 1039-1051 | 65.1 / 67.9 | within noise of baseline |
| `--spec-draft-backend-sampling` | — | — | already enabled by default in this build |

TG run-to-run variance on this rig is ~7 % (repeated baseline runs spanned 61.5-65.6 tok/s); treat
anything inside that as noise. `--backend-sampling` (`-bs`) read 67.9 on one run and 65.1 on a
repeat against a 65.6 baseline — noise, not a gain.

Note the `-ub` result: at 262144 the headroom budget is entirely spent on KV, so you **cannot buy
prompt-processing speed with a bigger ubatch**. If you want `-ub 1024` you must pay for it with
context (~229376), which is a bad trade for +3 % PP.

### `-cram`: the biggest agentic win

`--cache-ram` / `-cram` keeps evicted slot KV in **host** RAM so returning to an earlier
conversation restores instead of re-processing. Measured with `-np 1`: send a 45k-token prompt A,
then a different prompt B (evicting A), then A again — A returns in **0.9 s instead of 43 s**,
with `prompt eval time = ... / 4 tokens`.

The default is already 8192 MiB, which is only ~1 full 262k conversation (q4_0 KV at 262144 is
~7.2 GB). Switching between two long agentic sessions thrashes it. `-cram 32768` holds ~4. Size it
against host RAM, not VRAM.

### Reasoning effort dominates everything else

Qwen3.8 is a thinking model and **its chat template defaults to `xhigh`** — the highest supported
effort — whenever no reasoning flag is supplied. Straight from the GGUF's own template:

```jinja
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort == 'high' %}
        {%- set resolved_reasoning_effort = 'xhigh' %}
    {%- endif %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ...') }}
```

Note the second branch: **`high` is silently promoted to `xhigh`**, so asking for "high" does not
lower effort. Supported values are exactly `xhigh` (default), `medium`, `low` — anything else,
including `minimal`, raises from the template and surfaces as **HTTP 500**.

Measured on the same 22k agentic prompt, `-c 262144` config, `max_tokens: 4000`:

| Setting | Wall clock | Completion tokens | Answer |
|---|---|---|---|
| *(none supplied)* -> `xhigh` | 65 s | 4000 (capped) | **none — still thinking when it hit the cap** |
| `reasoning_effort: "low"` | 27.9 s | 1657 | complete |
| `reasoning_effort: "medium"` | 17.1 s | 1148 | complete |
| `chat_template_kwargs: {"enable_thinking": false}` | 13.4 s | 547 | complete |

**~5x wall clock, and the default did not answer at all.** No serving flag in this document comes
close; this is the single biggest lever for agentic latency. Because the default is the *maximum*
effort level, every request that omits `reasoning_effort` opts into the slowest mode by accident.

Set it per request (OpenAI-compatible `reasoning_effort`), since different agentic steps want
different depth. To pin a server-side default instead, use `--reasoning-effort <level>` or guard
against runaway with `--reasoning-budget N`. The served configs deliberately leave this unset so
clients keep control — decide per deployment.

### Client-side reasoning control (verified end-to-end through llama-swap)

The served configs leave reasoning **unset on purpose** so each client picks its own depth —
`xhigh` for coding agents, off for voice assistants. Everything below was tested against
`http://localhost:8080/v1/chat/completions`:

| Request field | Result |
|---|---|
| `"reasoning_effort": "xhigh"` | full effort — identical to sending nothing |
| `"reasoning_effort": "high"` | ⚠️ **silently promoted to `xhigh`** — does not reduce effort |
| `"reasoning_effort": "medium"` | reduced |
| `"reasoning_effort": "low"` | minimal thinking, still answers |
| `"reasoning_effort": "none"` | ✅ **thinking fully off** (llama.cpp intercepts this before the template) |
| `"reasoning_effort": "minimal"` | ❌ **HTTP 500** — template raises, not a supported level |
| `"chat_template_kwargs": {"enable_thinking": false}` | thinking fully off — equivalent to `none` |
| `"reasoning_budget": 0` | ⚠️ **silently ignored** — model still thinks |
| `"thinking": {"type": "disabled"}` | ⚠️ **silently ignored** — model still thinks |

**Use `reasoning_effort` and nothing else.** It is the OpenAI-standard field, so most clients
expose it, and `"none"` covers the full-off case without needing `chat_template_kwargs` support.
The two ignored fields are the dangerous ones: they look like they worked, return 200, and quietly
bill you a full `xhigh` think.

Safe value set for this model: **`none`, `low`, `medium`, `xhigh`**.

Per-client notes:

- **opencode** — set `reasoningEffort` in the provider/model `options` block. Its menu offers
  `none｜minimal｜low｜medium｜high｜xhigh`; of those, `minimal` returns HTTP 500 and `high` is a
  silent no-op. Known upstream bug: if the provider name contains a dot (e.g. `example.com`),
  model-level options are **silently dropped** ([opencode#23622]) — name the provider without dots.
- **Home Assistant** — the llama.cpp / OpenAI conversation integrations expose a thinking toggle
  and feed `reasoning_content` back as reasoning on 2026.4+. If your integration cannot send
  `reasoning_effort`, prefer one that can rather than reaching for `reasoning_budget` — that field
  is ignored here.
- **Any client that cannot set custom body fields** — there is no prompt-text escape hatch. This
  template has no `/no_think` trigger (the `/think` strings in it are just `</think>` tags), so a
  system-prompt instruction will not disable thinking. Serve a second llama-swap entry with
  `--reasoning-effort none` for that client instead; note llama-swap will unload/reload the model
  when switching between entries.

[opencode#23622]: https://github.com/anomalyco/opencode/issues/23622

### Stacking `ngram-mod` with `draft-mtp`

`--spec-type` takes a comma-separated list. `draft-mtp,ngram-mod` runs both drafters; upstream
notes they draft *independently* rather than pipelining ([#23184], closed as not planned), so the
gain is modest — but it is real and it costs no VRAM:

| Workload | `draft-mtp` | `draft-mtp,ngram-mod` |
|---|---|---|
| Freeform prose | 56.9 tok/s | 56.7 tok/s |
| Verbatim code reproduction | 65.0 tok/s | 69.9 tok/s |
| Realistic agentic turn (22k ctx → diff) | 60.3 tok/s | 66.8 tok/s |

`ngram-mod` is a **lossy hash table**, not a substring index — worth understanding before tuning
it (`common/ngram-mod.{h,cpp}`). It folds the last `--spec-ngram-mod-n-match` **tokens** (default
24, not characters) into one hash over a 4M-slot `int32` table and stores only the single next
token. Nothing is verified on lookup: keys are never stored, so a collision silently returns a
wrong token. That is safe because speculative decoding validates every draft token against the
real model — a bad guess is just a rejected draft. Drafting is also iterative: each hit is
appended and re-hashed for the next token, so it never needs one long span to match, and a
24-token window slides along any quoted region. Upstream warns below `n_match=16`; empirically
12 was worse than the default 24 here (67.1 vs 69.9 tok/s), so leave it alone.

Costs 16 MB of **host** RAM (4M × 4 B), zero VRAM. Two upstream caveats that did not reproduce on
`6fe7498`: [#23154] (CUDA OOM with this combo) and [#24507] (only one `spec-type` survives in
llama-server *router* mode — irrelevant here, llama-swap passes real CLI flags).

[#23184]: https://github.com/ggml-org/llama.cpp/issues/23184
[#23154]: https://github.com/ggml-org/llama.cpp/issues/23154
[#24507]: https://github.com/ggml-org/llama.cpp/issues/24507

### Leave ~1.4 GB of VRAM headroom — "it loads" is not "it works"

The binding constraint at long context is **not** the load-time footprint. KV and weights are
allocated up front, but prompt processing needs a transient compute buffer on top of them. Starve
it and the driver silently spills to host memory over PCIe: the model still loads, generation
still looks plausible, but **prompt processing collapses and gets worse the longer the prompt**,
with `nvidia-smi` showing 0 % GPU utilisation while it "works".

Measured on a 45,020-token prompt, same model, same flags, only KV precision changed:

| Config | VRAM used | Headroom | Prompt processing |
|---|---|---|---|
| `q4_0` K+V, `c 262144` | 23,120 MiB | 1,456 MiB | **1,042 tok/s** ✅ |
| `q8_0` K / `q4_0` V, `c 262144` | 23,926 MiB | 650 MiB | **84 → 43 → 23 tok/s** ❌ |

The second config loads without complaint and passes a short smoke test. On a real 227k prompt it
degrades to ~11 tok/s and takes tens of minutes. **Do not tune headroom below ~1.4 GB on a 24 GB
card**, and always validate a new config with a long prompt, never a short one — the failure is
invisible to short prompts because the compute buffer only grows with batch depth.

### KV cache precision at long context

`q4_0` K+V is what makes 262144 fit *with working headroom*, and it holds up under a hard eval —
not just a needle probe. `scripts/longctx_eval.py` builds a 6-task probe at **228,033 tokens**:

| Task | Result |
|---|---|
| 3-hop chain + arithmetic (`A7=3391` -> copied to `K2` -> doubled -> `M9`) | ✅ 6782 |
| 4-way distractor discrimination (Tessellate/Meridian/Halcyon/Peridot constants) | ✅ 74.219 |
| Exact alphanumeric ID (`7QX-40277-BJ19`) | ✅ |
| Supersession chain (12.5 -> 9.75 -> **8.25** bar; must return the last) | ✅ 8.25 |
| Two-point comparison across 44 % of the context | ✅ sector 811 |
| Exact float recall | ✅ 2.118 |

**6/6.** Run it with `python3 scripts/longctx_eval.py` then pipe the reply through
`scripts/longctx_grade.py`. This validates the vision config's shorter 196k window by extension.

If you still want a better-conditioned K cache, raise K and **give up context to keep the
headroom** — never spend the headroom itself:

```json
{"cache-type-k": "q8_0", "cache-type-v": "q4_0", "c": 229376}
```

22,934 MiB used, 1,642 MiB headroom — more margin than the working `q4_0` config. 229k instead of
262k is the price. (`q8_0` K at the full 262144 *does* load, at 23,926 MiB — and then prompt
processing collapses. See the headroom section above.)

### How this compares to what others report

Context for the numbers above (checked 2026-08-29):

- **Baseline llama.cpp on a 3090** is reported around 40 tok/s for this model. The 62-66 tok/s
  here is roughly 1.6x that, and consistent with [sudoingX/qwen38-mtp]'s +33 % MTP figure for a
  3090 (31.0 -> 41.3) plus the KV/context work in this document.
- **The widely-quoted 136.7 tok/s figure is an RTX 5090**, not a 3090 — see [KGP Talkie]. Same
  model and similar flags, ~2x the memory bandwidth. Not a target this card can reach.
- **vLLM can beat llama.cpp substantially on this exact GPU.** [syv-ai/qwen38-27b-rtx3090] reports
  ~114 tok/s single-user (~124 greedy) at 150k-262k context on one 3090, ~1.8x this setup. It gets
  there with a different stack: vLLM 0.27.1 plus custom patches, W4A16 AutoRound weights, int8
  activations, fp16 DeltaNet recurrent state, an int4 lm_head, and split-KV verify attention.
  Costs: custom patched install, no vision/mmproj path, and KV quantisation whose quality at depth
  the author marks unmeasured. A real option if raw single-user decode is the priority; a large
  migration away from the llama-swap/GGUF workflow this repo is built on.

One published recommendation conflicts with the measurements here: [KGP Talkie] lists `ngram-mod`
as not recommended. That was measured on prose at 32k on a 5090; on repetitive agentic work at
262k on a 3090 it is worth +7-11 % (table above). Prefer the local measurement for this workload.

[sudoingX/qwen38-mtp]: https://github.com/sudoingX/qwen38-mtp
[KGP Talkie]: https://kgptalkie.com/tutorials/generative-ai/qwen-3-8-27b-llama-cpp-speed-settings
[syv-ai/qwen38-27b-rtx3090]: https://github.com/syv-ai/qwen38-27b-rtx3090


## CSS (no Node.js)

```bash
./tailwindcss-linux-x64 -i ./input.css -o ./static/output.css --minify
```

## Development

Modular Python backend (`src/backend/`) with vanilla HTML/CSS/JS frontend (`src/frontend/`).

### Development Server

```bash
./dev.sh dev     # Start with source mounts + auto-reload
./dev.sh dev-down # Stop dev server
```

Source files are selectively mounted via `docker-compose.dev.yml` (preserves compiled CSS).

### Tests

139 tests across 9 test files. Run locally without Docker:

```bash
./dev.sh test
```

Covers: backend (config, models, state, cache, hf_hub, sync, websocket, api) + frontend (HTML structure, form elements, API routes, styling, layout, WebSocket).

### Frontend Structure

The frontend SPA has been modularized from a single ~700-line inline script into 9 separate JS modules:

| Module | Purpose |
|---|---|
| `state.js` | Shared application state |
| `utils.js` | DOM helpers, debounce, JSON validation |
| `api.js` | API communication functions |
| `filter.js` | Client-side search/filter |
| `localmodels.js` | Disk storage data fetching |
| `render.js` | UI rendering (cards, served symlink paths, progress bars) |
| `ws.js` | WebSocket connection management |
| `form.js` | Form handling, validation, mmproj + MTP head wiring |
| `app.js` | Main entry point, wires modules together |

### Source Structure

| File | Purpose |
|---|---|
| `src/backend/app.py` | FastAPI setup, lifespan, route definitions |
| `src/backend/api.py` | All route handlers |
| `src/backend/config.py` | Constants, paths, regex patterns |
| `src/backend/models.py` | Pydantic request/response models |
| `src/backend/state.py` | State file read/write |
| `src/backend/cache.py` | HF cache scanning and pruning |
| `src/backend/hf_hub.py` | HuggingFace API calls |
| `src/backend/sync.py` | llama-swap config sync |
| `src/backend/download.py` | Model downloads (disk-based progress, stall watchdog, resume) |
| `src/backend/websocket.py` | WebSocket connection manager |
| `src/frontend/index.html` | SPA HTML (forms, layout, styling) |
| `src/frontend/js/*.js` | 9 modular JavaScript files |
