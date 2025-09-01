# Persona Builder CLI

## Overview
- Ingests GA4 + Shopify cohort stats and emits production-ready YAML personas.
- Applies intent labeling, clipping, dwell scaling, backtrack/form error probabilities, and cohort weighting.
- Merges tiny cohorts into representative personas to reduce sparsity.

## Input
- Accepts either:
  - A list/dict of cohort objects, or
  - A GA4-style wrapper: `{ global_median_dwell_sec, window_days, cohorts: [ { key: {device, source, country, session_depth}, ... } ] }`.
- Cohort fields (synonyms supported):
  - `device`, `source`, `country`, `session_depth`
  - `sessions`, `cr`, `bounce_rate`
  - `dwell_means` | `dwell_mean` | `dwell_mean_sec`
  - `backtrack_rate`, `form_error_rate`
- Normalization:
  - `device` → `mobile|desktop` (defaults to `desktop` if unknown)
  - `source` → `ads|organic|direct|referral` (heuristics for ad/referral terms)

## Rules Applied
- Intent: cold/warm/hot via tertiles over `cr` (higher is better) and `bounce_rate` (lower is better).
- buy_propensity = `cr` clipped to [0.005, 0.20].
- dwell_scale = `dwell_mean` / `global_median_dwell` clipped to [0.8, 1.5]. Uses `global_median_dwell_sec` if provided; otherwise median of cohorts.
- backtrack_p = `backtrack_rate` clipped to [0.05, 0.35].
- form_error_p = `form_error_rate` clipped to [0.02, 0.20].
- weight = `sessions` / sum(`sessions`).

## Merging Small Cohorts
- Cohorts with `sessions < 500` (configurable via `--min-sessions`) are merged by `(device, source, intent)` using session-weighted averages.

## Usage
- From file: `python src/personas_cli.py -i cohorts.json > personas.yaml`
- From stdin: `cat cohorts.json | python src/personas_cli.py > personas.yaml`
- Tune merge threshold: `python src/personas_cli.py -i cohorts.json --min-sessions 300`

## Output Schema
```yaml
personas:
  <persona_id>:
    device: mobile|desktop
    source: ads|organic|direct|referral
    intent: cold|warm|hot
    buy_propensity: float
    dwell_scale: float
    backtrack_p: float
    form_error_p: float
    weight: float
```

Example persona block:
```yaml
personas:
  mobile_ads_cold_1:
    device: mobile
    source: ads
    intent: cold
    buy_propensity: 0.026000
    dwell_scale: 1.183333
    backtrack_p: 0.200000
    form_error_p: 0.090000
    weight: 0.161789
```
