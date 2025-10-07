# SeeAct2 API Documentation

## Overview

The SeeAct2 API is a FastAPI-based backend service that provides endpoints for managing experiments, calibrations, and personas. The API is designed to support A/B testing workflows with synthetic user simulation capabilities.

## Base URL

```
http://localhost:8001
```

## Authentication

All API endpoints require the `X-User-ID` header to identify the user making the request. Some endpoints may require additional authentication headers in the future.

### Required Headers

- `X-User-ID`: User identifier (required for most endpoints)
- `Content-Type: application/json` (for POST requests)
- `Idempotency-Key`: Optional, for ensuring idempotent operations

## API Endpoints

### 1. Experiments API (`/v1/experiments`)

The Experiments API manages A/B testing experiments with synthetic user simulation.

#### 1.1 Create Experiment

**Endpoint:** `POST /v1/experiments` or `POST /v1/experiments/`

**Description:** Creates a new A/B testing experiment with synthetic user simulation.

**Headers:**

- `X-User-ID`: Required
- `Idempotency-Key`: Optional (recommended for retry safety)

**Request Body:**

```json
{
  "user_id": "string",
  "site_id": "string",
  "name": "string",
  "variant_a_url": "string",
  "variant_b_url": "string",
  "n_agents": 100,
  "calibration_id": "string",
  "concurrency": 10,
  "provider": "browserbase",
  "model": "gpt-4o",
  "max_cost_usd": 50.0,
  "seed": 12345,
  "min_per_arm": 200,
  "min_completion_ratio": 0.8
}
```

**Response:**

```json
{
  "experiment_id": "uuid",
  "status": "queued"
}
```

**Status Codes:**

- `200`: Experiment created successfully
- `400`: Bad request (missing required fields)
- `500`: Internal server error

#### 1.2 Get Experiment Status

**Endpoint:** `GET /v1/experiments/{experiment_id}`

**Description:** Retrieves detailed status and metrics for a specific experiment.

**Headers:**

- `X-User-ID`: Required

**Response:**

```json
{
  "experiment_id": "uuid",
  "user_id": "string",
  "site_id": "string",
  "name": "string",
  "status": "running",
  "variant_a_url": "string",
  "variant_b_url": "string",
  "n_agents": 100,
  "concurrency": 10,
  "provider": "browserbase",
  "model": "gpt-4o",
  "max_cost_usd": 50.0,
  "seed": 12345,
  "min_per_arm": 200,
  "min_completion_ratio": 0.8,
  "started_at": "2024-01-01T00:00:00Z",
  "finished_at": null,
  "winner": null,
  "lift_abs": null,
  "lift_rel": null,
  "p_value": null,
  "aggregates": {
    "A": {
      "n": 50,
      "conversions": 5,
      "conversion_rate": 0.1
    },
    "B": {
      "n": 50,
      "conversions": 8,
      "conversion_rate": 0.16
    }
  },
  "partial_metrics": {...},  // Only for running experiments
  "partial_summary": {...}   // Only for running experiments
}
```

#### 1.3 List Experiments

**Endpoint:** `GET /v1/experiments` or `GET /v1/experiments/`

**Description:** Lists experiments with optional status filtering.

**Headers:**

- `X-User-ID`: Required

**Query Parameters:**

- `status`: Optional filter (`in_progress`, `ended`)

**Response:**

```json
{
  "experiments": [
    {
      "id": "uuid",
      "user_id": "string",
      "name": "string",
      "site_id": "string",
      "date": "2024-01-01",
      "status": "running",
      "lift_rel": 0.6,
      "result": "A"
    }
  ],
  "total": 1
}
```

#### 1.4 Get Experiment Events (SSE)

**Endpoint:** `GET /v1/experiments/{experiment_id}/events`

**Description:** Streams real-time events for an experiment using Server-Sent Events.

**Headers:**

- `Authorization`: Optional (future use)

**Response:** Server-Sent Events stream

```
event: experiment_started
data: {"experiment_id": "uuid", "timestamp": "2024-01-01T00:00:00Z"}

event: agent_completed
data: {"agent_id": "agent_1", "variant": "A", "conversion": true}

event: experiment_completed
data: {"experiment_id": "uuid", "winner": "B", "lift_rel": 0.6}
```

#### 1.5 Get Experiment Artifacts

**Endpoint:** `GET /v1/experiments/{experiment_id}/artifacts`

**Description:** Retrieves URLs to experiment artifacts (screenshots, videos, etc.).

**Response:**

```json
{
  "experiment_id": "uuid",
  "artifacts": {
    "screenshots": ["url1", "url2"],
    "videos": ["url1"],
    "logs": ["url1"]
  }
}
```

#### 1.6 Get Experiment Summary (CSV)

**Endpoint:** `GET /v1/experiments/{experiment_id}/summary.csv`

**Description:** Downloads experiment summary as CSV file.

**Response:** CSV file download

#### 1.7 Get Variant Data

**Endpoints:**

- `GET /v1/experiments/{experiment_id}/A.json`
- `GET /v1/experiments/{experiment_id}/B.json`

**Description:** Retrieves detailed data for specific variants.

**Response:**

```json
{
  "experiment_id": "uuid",
  "variant": "A",
  "url": "https://example.com/variant-a",
  "aggregates": {
    "n": 50,
    "conversions": 5,
    "conversion_rate": 0.1
  },
  "sessions": []
}
```

#### 1.8 Get Experiment Metrics

**Endpoint:** `GET /v1/experiments/{experiment_id}/metrics`

**Description:** Retrieves detailed metrics for the experiment details page.

**Headers:**

- `X-User-ID`: Required

**Response:**

```json
{
  "experiment_id": "uuid",
  "metrics": {
    "total_visits": 100,
    "total_conversions": 13,
    "overall_conversion_rate": 0.13,
    "variant_a": {
      "visits": 50,
      "conversions": 5,
      "conversion_rate": 0.1
    },
    "variant_b": {
      "visits": 50,
      "conversions": 8,
      "conversion_rate": 0.16
    }
  }
}
```

#### 1.9 Get Experiment Summary

**Endpoint:** `GET /v1/experiments/{experiment_id}/summary`

**Description:** Retrieves experiment summary for the details page.

**Headers:**

- `X-User-ID`: Required

**Response:**

```json
{
  "experiment_id": "uuid",
  "summary": {
    "winner": "B",
    "lift_abs": 0.06,
    "lift_rel": 0.6,
    "p_value": 0.05,
    "confidence_level": 0.95
  }
}
```

#### 1.10 Delete Experiment

**Endpoint:** `DELETE /v1/experiments/{experiment_id}`

**Description:** Deletes an experiment and all associated data.

**Headers:**

- `X-User-ID`: Required

**Response:**

```json
{
  "message": "Experiment deleted successfully"
}
```

### 2. Calibrations API (`/v1/calibrations`)

The Calibrations API manages the calibration process that analyzes real user behavior to create synthetic personas.

#### 2.1 List Calibrations

**Endpoint:** `GET /v1/calibrations` or `GET /v1/calibrations/`

**Description:** Lists calibrations for a specific user and site.

**Headers:**

- `X-User-ID`: Required

**Query Parameters:**

- `site_id`: Required

**Response:**

```json
[
  {
    "calibration_id": "uuid",
    "user_id": "string",
    "site_id": "string",
    "status": "complete",
    "steps": [
      {
        "name": "fetch_ga_snapshot_neon",
        "status": "complete"
      },
      {
        "name": "generate_prompts",
        "status": "complete"
      }
    ],
    "metrics": {
      "num_prompts": 100,
      "num_features": 7
    },
    "started_at": "2024-01-01T00:00:00Z",
    "finished_at": "2024-01-01T01:00:00Z",
    "error": null
  }
]
```

#### 2.2 Start Calibration

**Endpoint:** `POST /v1/calibrations` or `POST /v1/calibrations/`

**Description:** Starts a new calibration process for a site.

**Headers:**

- `X-User-ID`: Required

**Request Body:**

```json
{
  "site_id": "string",
  "seed": 12345
}
```

**Response:**

```json
{
  "calibration_id": "uuid",
  "status": "queued"
}
```

#### 2.3 Get Calibration Status

**Endpoint:** `GET /v1/calibrations/{calibration_id}`

**Description:** Retrieves the current status of a calibration process.

**Response:**

```json
{
  "calibration_id": "uuid",
  "user_id": "string",
  "site_id": "string",
  "status": "running",
  "steps": [
    {
      "name": "fetch_ga_snapshot_neon",
      "status": "complete"
    },
    {
      "name": "generate_prompts",
      "status": "running"
    }
  ],
  "metrics": {
    "num_prompts": 50
  },
  "started_at": "2024-01-01T00:00:00Z",
  "finished_at": null,
  "error": null
}
```

#### 2.4 Get Calibration Features

**Endpoint:** `GET /v1/calibrations/{calibration_id}/features`

**Description:** Retrieves calibration features (user behavior distributions).

**Response:**

```json
{
  "distributions": [
    {
      "dimension": "device",
      "real": { "mobile": 0.6, "desktop": 0.4 },
      "synthetic": { "mobile": 0.58, "desktop": 0.42 }
    }
  ]
}
```

#### 2.5 Get Calibration Behavior Match

**Endpoint:** `GET /v1/calibrations/{calibration_id}/behavior-match`

**Description:** Retrieves behavior match data (conversion funnel rates).

**Response:**

```json
{
  "real": [
    {
      "event": "page_view",
      "rate": 1.0
    },
    {
      "event": "add_to_cart",
      "rate": 0.15
    }
  ],
  "synthetic": [
    {
      "event": "page_view",
      "rate": 1.0
    },
    {
      "event": "add_to_cart",
      "rate": 0.14
    }
  ]
}
```

#### 2.6 Get Calibration Events (SSE)

**Endpoint:** `GET /v1/calibrations/{calibration_id}/events`

**Description:** Streams real-time events for a calibration process.

**Response:** Server-Sent Events stream

```
event: queued
data: {"calibration_id": "uuid"}

event: ga_snapshot_neon_complete
data: {"calibration_id": "uuid"}

event: prompts_generated
data: {"count": 100}

event: features_real_ready
data: {"calibration_id": "uuid"}

event: complete
data: {"calibration_id": "uuid"}
```

### 3. Personas API (`/v1/personas`)

The Personas API manages synthetic user personas for A/B testing.

#### 3.1 Generate Master Persona Pool

**Endpoint:** `POST /v1/personas/generate-master`

**Description:** Generates a master pool of synthetic personas based on real user data.

**Request Body:**

```json
{
  "window_days": 30,
  "window_end": "2024-01-01T00:00:00Z",
  "min_sessions": 5,
  "k_anonymity": 50,
  "unknown_drop_threshold": 0.7,
  "conversion_events": ["purchase", "add_to_cart"],
  "include_events_extra": ["page_view"],
  "dsn": "postgresql://...",
  "include_prompts": true,
  "include_summary": true,
  "persist_db": true,
  "persist_local": true,
  "site_domain": "example.com"
}
```

**Response:**

```json
{
  "pool_id": "sha1_hash",
  "count": 1000,
  "window_end": "2024-01-01T00:00:00Z",
  "persisted_db": true,
  "persisted_local": true,
  "artifacts": {
    "master_pool.jsonl": "path/to/file",
    "summary.json": "path/to/file"
  },
  "summary": {
    "pool_id": "sha1_hash",
    "distributions": {
      "real": {...},
      "synthetic": {...}
    },
    "behavior": {
      "real": {...},
      "synthetic": {...}
    }
  }
}
```

#### 3.2 List Personas

**Endpoint:** `GET /v1/personas/`

**Description:** Lists available personas from the master pool.

**Query Parameters:**

- `limit`: Number of personas to return (1-500, default: 50)
- `offset`: Number of personas to skip (default: 0)

**Response:**

```json
{
  "total": null,
  "items": [
    {
      "persona_id": "uuid",
      "device": "mobile",
      "source": "organic",
      "operatingSystem": "iOS",
      "userAgeBracket": "25-34",
      "newVsReturning": "new",
      "gender": "female",
      "geo": "US",
      "intent": "browse",
      "metrics": {
        "sessions": 5,
        "page_views": 12,
        "conversions": 1
      },
      "weight": 0.8
    }
  ]
}
```

#### 3.3 Sample Personas

**Endpoint:** `POST /v1/personas/sample`

**Description:** Samples personas from the master pool using various strategies.

**Request Body:**

```json
{
  "size": 10,
  "strategy": "weighted",
  "persona_ids": ["uuid1", "uuid2"]
}
```

**Response:**

```json
[
  {
    "persona_id": "uuid",
    "device": "mobile",
    "source": "organic",
    "operatingSystem": "iOS",
    "userAgeBracket": "25-34",
    "newVsReturning": "new",
    "gender": "female",
    "geo": "US",
    "intent": "browse",
    "metrics": {
      "sessions": 5,
      "page_views": 12,
      "conversions": 1
    },
    "weight": 0.8
  }
]
```

#### 3.4 Get Traffic Summary

**Endpoint:** `GET /v1/personas/traffic-summary`

**Description:** Retrieves traffic distribution summary (real vs synthetic).

**Response:**

```json
{
  "real": {
    "device": { "mobile": 0.6, "desktop": 0.4 },
    "source": { "organic": 0.4, "direct": 0.3, "social": 0.3 },
    "operatingSystem": { "iOS": 0.4, "Android": 0.3, "Windows": 0.3 }
  },
  "synthetic": {
    "device": { "mobile": 0.58, "desktop": 0.42 },
    "source": { "organic": 0.41, "direct": 0.29, "social": 0.3 },
    "operatingSystem": { "iOS": 0.39, "Android": 0.31, "Windows": 0.3 }
  }
}
```

#### 3.5 Get Behavior Match

**Endpoint:** `GET /v1/personas/behavior-match`

**Description:** Retrieves behavior match data (conversion rates).

**Response:**

```json
{
  "real": {
    "page_view": 1.0,
    "add_to_cart": 0.15,
    "purchase": 0.05
  },
  "synthetic": {
    "page_view": 1.0,
    "add_to_cart": 0.14,
    "purchase": 0.05
  }
}
```

#### 3.6 Scrape Site Vocabulary

**Endpoint:** `POST /v1/personas/scrape-vocab`

**Description:** Scrapes a Shopify site to extract vocabulary and product information.

**Request Body:**

```json
{
  "site": "example.myshopify.com",
  "max_pages": 50,
  "user_agent": "SeeAct2Bot/1.0",
  "persist": true,
  "dsn": "postgresql://..."
}
```

**Response:**

```json
{
  "vocab": {
    "site_domain": "example.myshopify.com",
    "products": [...],
    "categories": [...],
    "brands": [...],
    "keywords": [...]
  },
  "persisted": true
}
```

#### 3.7 Generate Prompts

**Endpoint:** `POST /v1/personas/generate-prompts`

**Description:** Generates AI prompts for specific personas.

**Request Body:**

```json
{
  "persona_ids": ["uuid1", "uuid2"],
  "temperature": 0.4,
  "regenerate": false,
  "site_domain": "example.com",
  "include_vocab": true,
  "out_dir": "/path/to/output",
  "dsn": "postgresql://..."
}
```

**Response:**

```json
{
  "count": 2,
  "site_domain": "example.com",
  "items": [
    {
      "persona_id": "uuid1",
      "path": "/path/to/shop_prompt_uuid1.py"
    }
  ]
}
```

#### 3.8 Get Persona Prompt

**Endpoint:** `GET /v1/personas/{persona_id}/prompt`

**Description:** Retrieves the generated prompt for a specific persona.

**Query Parameters:**

- `site_domain`: Optional site domain

**Response:**

```json
{
  "persona_id": "uuid",
  "site_domain": "example.com",
  "prompt": "You are a 25-34 year old female from the US..."
}
```

## API Flow Examples

### Complete A/B Testing Workflow

1. **Generate Personas** (if not already done):

   ```bash
   POST /v1/personas/generate-master
   ```

2. **Start Calibration**:

   ```bash
   POST /v1/calibrations
   ```

3. **Wait for Calibration to Complete** (monitor via SSE):

   ```bash
   GET /v1/calibrations/{calibration_id}/events
   ```

4. **Create Experiment**:

   ```bash
   POST /v1/experiments
   ```

5. **Monitor Experiment Progress** (via SSE):

   ```bash
   GET /v1/experiments/{experiment_id}/events
   ```

6. **Get Results**:
   ```bash
   GET /v1/experiments/{experiment_id}
   GET /v1/experiments/{experiment_id}/metrics
   GET /v1/experiments/{experiment_id}/summary
   ```

### Real-time Monitoring

Both experiments and calibrations support Server-Sent Events (SSE) for real-time monitoring:

```javascript
const eventSource = new EventSource("/v1/experiments/experiment-id/events");
eventSource.onmessage = function (event) {
  const data = JSON.parse(event.data);
  console.log("Event received:", data);
};
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (missing required fields, invalid data)
- `401`: Unauthorized (missing or invalid authentication)
- `404`: Not Found (resource doesn't exist)
- `500`: Internal Server Error

Error responses include a JSON body with error details:

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Rate Limiting

Currently, there are no rate limits implemented, but they may be added in future versions.

## CORS

The API supports CORS for cross-origin requests from web applications. All origins are currently allowed (`*`), but this should be restricted in production environments.

## Database

The API uses PostgreSQL (Neon Database) for persistent storage. Connection pooling is implemented for optimal performance.

## Performance Optimizations

- **Connection Pooling**: Database connections are pooled for better performance
- **Caching**: In-memory caching is implemented for frequently accessed data
- **Optimized Queries**: SQL queries are optimized to avoid N+1 problems
- **Background Processing**: Long-running tasks are processed in the background

## Health Checks

Each service provides a health check endpoint:

- Experiments: `GET /v1/experiments/health`
- Calibrations: `GET /v1/calibrations/health`
- Personas: No specific health endpoint (uses main app health)

## Development

To run the API locally:

```bash
cd SeeAct2
source venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload
```

The API will be available at `http://localhost:8001` with automatic API documentation at `http://localhost:8001/docs`.
