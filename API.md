# Squoosh API v1

This document describes the HTTP APIs that power Calibration and Experiment runs for synthetic-agent testing. It complements the CLI flows in the README and is additive — the CLI remains supported and documented.

- Base URL: https://www.squoosh.ai
- Auth: `Authorization: Bearer <token>` (tenancy inferred from the token)
- Site scoping: Every request includes a `site_id` that must belong to the caller’s org
- Versioning: Path-based only (e.g., `/v1/...`) — no org/tenant prefix
- Changelog: [TODO] See README “Changelog” section; add entries on breaking changes
- Runtime manifests now live on disk. The runner and CLI read JSON files from a manifest directory (default: `site_manifest/` at repo root; override with `SEEACT_MANIFEST_DIR` or `--manifest-dir`). The old Python manifest package from earlier releases has been removed.
- A sample manifest (`site_manifest/example.com.json`) ships with the repo. A fresh environment can verify installation via:
  ```bash
  python -m seeact.runner --tasks data/online_tasks/sample_tasks.json --metrics-dir runs/smoke
  ```
  (expects a quick smoke run; set `SEEACT_MANIFEST_DIR` first if you relocate manifests.)

## Personas‑only API (existing)
The personas endpoints used by the CLI remain unchanged. See the README section “API Reference (Personas‑only)” for details. The Calibration Job API orchestrates those endpoints under a single call and streams progress over SSE.

---

## Calibration Job API
One‑click wrapper for the 7‑step Calibrate UX with live progress via SSE.

### Overview {#calibration-job-overview}
- Why: Personas endpoints exist, but UX needs a single call + streaming progress for the “Calibrate” button.
- Data source (v1): GA‑derived traffic is pulled from Neon Postgres (not GA4 API). `site_id` selects the site’s latest traffic window.
- Implementation scaffolding [TODO]:
  - GA–Neon adapter that reads normalized GA traffic & funnel metrics from Neon
  - Env vars: `NEON_DATABASE_URL` (required), `NEON_SCHEMA` (optional, default `public`)
  - Expected tables (names TBD — follow current schema): `traffic_snapshots`, `funnel_metrics`

What it does (7 steps):
1) Fetch & snapshot Neon traffic for `site_id` (normalized + fingerprint), persist
2) Generate 1,000 prompts proportional to distributions; encode site purchase rate; persist
3) Write real traffic distributions (inner pie)
4) Write synthetic distributions (outer pie) aggregated from prompts
5) Write real funnel rates (six Shopify events)
6) Write synthetic target funnel rates (v1 equals real; later may derive from prompt metadata)
7) Emit SSE waypoints; mark complete

Idempotency: Send `Idempotency-Key`; server dedupes by snapshot fingerprint within 30 minutes per site.

### Endpoints {#calibration-job-endpoints}
- POST `/v1/calibrations`
  - Headers: `Authorization: Bearer <token>`, `Idempotency-Key: <uuid>` (recommended)
  - Body:
    ```json
    { "site_id": "hijabkart.in", "seed": 12345 }
    ```
  - 201:
    ```json
    { "calibration_id": "a0a1f2e4-5b6c-47d2-a3f9-20b7c3b1ad50", "status": "queued" }
    ```
- GET `/v1/calibrations/{calibration_id}`
  ```json
  { "status": "running", "steps": [
    {"name":"fetch_ga_snapshot_neon","status":"complete"},
    {"name":"generate_prompts","status":"running"}
  ], "metrics": {"num_prompts": 1000} }
  ```
- GET `/v1/calibrations/{calibration_id}/features`
  ```json
  { "distributions": [
    {"dimension":"source_medium","kind":"real","buckets":[{"bucket":"google / organic","pct":0.38},{"bucket":"direct","pct":0.44}]},
    {"dimension":"source_medium","kind":"synthetic","buckets":[{"bucket":"google / organic","pct":0.38},{"bucket":"direct","pct":0.44}]},
    {"dimension":"device_category","kind":"real","buckets":[{"bucket":"desktop","pct":0.59},{"bucket":"mobile","pct":0.41}]},
    {"dimension":"device_category","kind":"synthetic","buckets":[{"bucket":"desktop","pct":0.59},{"bucket":"mobile","pct":0.41}]},
    {"dimension":"geo_bucket","kind":"real","buckets":[{"bucket":"United States","pct":0.27},{"bucket":"India","pct":0.06}]},
    {"dimension":"geo_bucket","kind":"synthetic","buckets":[{"bucket":"United States","pct":0.27},{"bucket":"India","pct":0.06}]}
  ]}
  ```
- GET `/v1/calibrations/{calibration_id}/behavior-match`
  ```json
  { "real": [
    {"event":"page_view","rate":1.0},
    {"event":"view_item","rate":0.42},
    {"event":"view_item_list","rate":0.61},
    {"event":"add_to_cart","rate":0.09},
    {"event":"begin_checkout","rate":0.014},
    {"event":"purchase","rate":0.003}
  ], "synthetic": [
    {"event":"page_view","rate":1.0},
    {"event":"view_item","rate":0.42},
    {"event":"view_item_list","rate":0.61},
    {"event":"add_to_cart","rate":0.09},
    {"event":"begin_checkout","rate":0.014},
    {"event":"purchase","rate":0.003}
  ]}
  ```
  v1 sets synthetic == real. [TODO] Later derive synthetic targets from prompt metadata while keeping response shape.

### Event Stream (SSE) {#calibration-job-sse}
GET `/v1/calibrations/{calibration_id}/events` → `text/event-stream`
- Emits: `queued`, `ga_snapshot_neon_complete`, `prompts_generated {"count":1000}`, `features_real_ready`, `features_synth_ready`, `behavior_real_ready`, `behavior_synth_ready`, `complete`, `error {"message":"..."}`
- Keep‑alive: heartbeat comment every 15s: `:ka` (clients ignore comment frames)

### Data Model {#calibration-job-data-model}
- `calibrations(id uuid, site_id text, status text, started_at timestamptz, finished_at timestamptz, ga_snapshot_id uuid, seed int, idempotency_key text, fingerprint text, error text)`
- `ga_snapshots(id uuid, site_id text, taken_at timestamptz, window_start timestamptz, window_end timestamptz, payload_json jsonb, fingerprint text, purchase_rate float)` (Neon source in v1)
- `traffic_distributions(calibration_id uuid, dimension text, bucket text, kind text check(kind in ('real','synthetic')), pct float)`
- `event_rates(calibration_id uuid, event text, kind text check(kind in ('real','synthetic')), rate float)`
- `synthetic_prompts(id uuid, calibration_id uuid, persona_id text, source_medium text, device_category text, geo_bucket text, encoded_purchase_rate float, prompt_text text)`
- Privacy/bucket policy: replicate UXAgent bucket handling; no additional k‑anon beyond that. [TODO] Link to UXAgent references later.

### End‑to‑end Flow {#calibration-job-e2e}
1) POST `/v1/calibrations`
2) SSE: `queued` → `ga_snapshot_neon_complete`
3) SSE: `prompts_generated`
4) SSE: `features_real_ready` → draw inner pies
5) SSE: `features_synth_ready` → draw outer pies
6) SSE: `behavior_*_ready` → draw bars
7) SSE: `complete` → show “Calibrated”

### Examples (curl)
- Start a calibration
```bash
curl -X POST https://www.squoosh.ai/v1/calibrations \
  -H "Authorization: Bearer $TOKEN" \
  -H "Idempotency-Key: $(uuidgen)" \
  -H "Content-Type: application/json" \
  -d '{"site_id":"hijabkart.in","seed":12345}'
```
- Stream events (SSE)
```bash
curl -N https://www.squoosh.ai/v1/calibrations/<calibration_id>/events \
  -H "Authorization: Bearer $TOKEN"
```

---

## Experiments API
End‑to‑end A/B orchestration for calibrated synthetic sessions with live streaming and final stats.

### Overview {#experiments-overview}
- Inputs: A/B URLs, prior `calibration_id`; 50/50 assignment stratified by `(source_medium, device_category, geo_bucket)`
- Provider (v1): `browserbase` only — Playwright local/managed is [TODO]
- Winner defaults: `min_per_arm = 200`, `min_completion_ratio = 0.80` (override per request)
- Cost guardrail: `max_cost_usd` required; cancel remaining shards when exceeded [TODO]

### Endpoints {#experiments-endpoints}
- POST `/v1/experiments`
  - Headers: `Authorization: Bearer <token>`, `Idempotency-Key: <uuid>`
  - Body:
    ```json
    {
      "site_id": "hijabkart.in",
      "name": "Homepage hero test",
      "variant_a_url": "https://hijabkart.in/?v=a",
      "variant_b_url": "https://hijabkart.in/?v=b",
      "n_agents": 1000,
      "calibration_id": "a0a1f2e4-5b6c-47d2-a3f9-20b7c3b1ad50",
      "concurrency": 50,
      "provider": "browserbase",
      "model": "gpt-4o",
      "max_cost_usd": 50.0,
      "seed": 12345,
      "min_per_arm": 200,
      "min_completion_ratio": 0.8
    }
    ```
  - 201:
    ```json
    { "experiment_id": "2c4efb77-4f2d-4f03-9a5f-1c74d3b9a6d9", "status": "queued" }
    ```
- GET `/v1/experiments/{experiment_id}`
  ```json
  {
    "status": "running",
    "name": "Homepage hero test",
    "variant_a_url": "...",
    "variant_b_url": "...",
    "aggregates": {
      "A": { "finished": 231, "purchases": 7, "cr": 0.0303 },
      "B": { "finished": 228, "purchases": 11, "cr": 0.0482 }
    },
    "winner": null,
    "lift_rel": null
  }
  ```
- GET `/v1/experiments?status=in_progress|ended`
  ```json
  [
    {"id":"...","name":"...","date":"2025-09-20","lift_rel":0.443,"result":"Variant"}
  ]
  ```

### Event Stream (SSE) {#experiments-sse}
GET `/v1/experiments/{experiment_id}/events` → `text/event-stream`
- Events: `queued`, `agent_started {"session_id":"...","variant":"A"}`, `funnel_event {...}`, `progress {...}`, `complete {...}`, `error {...}`
- Keep‑alive: heartbeat comment every 15s: `:ka`

### Data Model {#experiments-data-model}
- `experiments(id uuid, site_id text, name text, status text, started_at timestamptz, finished_at timestamptz, variant_a_url text, variant_b_url text, n_agents int, seed int, provider text, model text, max_cost_usd numeric, result text, lift_abs numeric, lift_rel numeric, p_value numeric, error text)`
- `agent_sessions(id uuid, experiment_id uuid, prompt_id uuid, variant text check(variant in ('A','B')), status text, started_at timestamptz, finished_at timestamptz, purchased boolean, metrics_path text, events_jsonb jsonb)`
- `variant_metrics(experiment_id uuid, variant text, n int, purchases int, cr numeric, add_to_cart_rate numeric, begin_checkout_rate numeric, purchase_rate numeric)`
- `experiment_events(id uuid, experiment_id uuid, ts timestamptz, type text, payload_jsonb jsonb)` (optional, for audits/SSE replay)

### A/B Assignment & Stats {#experiments-ab-and-stats}
- Stratification keys: `(source_medium, device_category, geo_bucket)` (from calibration prompt rows)
- Deterministic split: e.g., `hash(prompt_id, seed) % 2` → A/B
- Denominator: CR uses finished sessions; track timeouts/errors separately
- Winner defaults: `min_per_arm = 200`, `min_completion_ratio = 0.80`
- Significance (default): two‑proportion z‑test (pooled). Winner = higher CR if thresholds met and `p ≤ 0.05`
- [TODO] Bayesian: Beta(1,1) priors; report `Pr(B>A)` with a documented threshold (e.g., ≥0.95)

### End‑to‑end Flow {#experiments-e2e}
1) POST `/v1/experiments` (A/B URLs + `calibration_id`)
2) SSE: `queued` → sessions created/sharded
3) Agents run; SSE emits `agent_started`, `funnel_event`, periodic `progress`
4) All sessions finish or timebox hits → compute per‑variant CR & stats → persist `variant_metrics` + `experiments.result`
5) SSE: `complete` (winner + p‑value) → UI shows Winner
6) Dashboard tabs: In‑Progress (`GET /v1/experiments?status=in_progress`) and Ended (`GET /v1/experiments?status=ended`)

### Examples (curl)
- Create an experiment
```bash
curl -X POST https://www.squoosh.ai/v1/experiments \
  -H "Authorization: Bearer $TOKEN" \
  -H "Idempotency-Key: $(uuidgen)" \
  -H "Content-Type: application/json" \
  -d '{
        "site_id":"hijabkart.in",
        "name":"Homepage hero test",
        "variant_a_url":"https://hijabkart.in/?v=a",
        "variant_b_url":"https://hijabkart.in/?v=b",
        "n_agents":1000,
        "calibration_id":"<from calibration>",
        "concurrency":50,
        "provider":"browserbase",
        "model":"gpt-4o",
        "max_cost_usd":50.0,
        "seed":12345,
        "min_per_arm":200,
        "min_completion_ratio":0.8
      }'
```
- Stream experiment events (SSE)
```bash
curl -N https://www.squoosh.ai/v1/experiments/<experiment_id>/events \
  -H "Authorization: Bearer $TOKEN"
```

---

## Auth, Idempotency, Limits (shared)
- All endpoints require `Authorization: Bearer <token>`
- Send `Idempotency-Key` on mutating routes for safe retries
- Rate limits: [TODO] per‑org ceilings (e.g., 3 concurrent experiments, 1 active calibration per site)
- Cost guardrail: [TODO] terminate remaining shards when estimated cost exceeds `max_cost_usd`

## Changelog [TODO]
Add release notes when breaking changes are introduced.
