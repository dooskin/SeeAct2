-- Database schema for SeeAct2 API
-- This file contains the SQL schema for all API tables

-- Calibration tables
CREATE TABLE IF NOT EXISTS calibrations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calibration_id TEXT UNIQUE NOT NULL,
    user_id TEXT NOT NULL,
    site_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'complete', 'error')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    ga_snapshot_id UUID,
    seed INTEGER,
    idempotency_key TEXT,
    fingerprint TEXT,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Calibration features table
CREATE TABLE IF NOT EXISTS calibration_features (
    calibration_id TEXT PRIMARY KEY REFERENCES calibrations(calibration_id) ON DELETE CASCADE,
    features JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Calibration behavior match table
CREATE TABLE IF NOT EXISTS calibration_behavior_match (
    calibration_id TEXT PRIMARY KEY REFERENCES calibrations(calibration_id) ON DELETE CASCADE,
    behavior_match JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ga_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    site_id TEXT NOT NULL,
    taken_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    payload_json JSONB NOT NULL,
    fingerprint TEXT NOT NULL,
    purchase_rate FLOAT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS traffic_distributions (
    calibration_id UUID NOT NULL REFERENCES calibrations(id) ON DELETE CASCADE,
    dimension TEXT NOT NULL,
    bucket TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('real', 'synthetic')),
    pct FLOAT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (calibration_id, dimension, bucket, kind)
);

CREATE TABLE IF NOT EXISTS event_rates (
    calibration_id UUID NOT NULL REFERENCES calibrations(id) ON DELETE CASCADE,
    event TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('real', 'synthetic')),
    rate FLOAT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (calibration_id, event, kind)
);

CREATE TABLE IF NOT EXISTS synthetic_prompts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calibration_id UUID NOT NULL REFERENCES calibrations(id) ON DELETE CASCADE,
    persona_id TEXT NOT NULL,
    source_medium TEXT NOT NULL,
    device_category TEXT NOT NULL,
    geo_bucket TEXT NOT NULL,
    encoded_purchase_rate FLOAT NOT NULL,
    prompt_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Experiment tables
CREATE TABLE IF NOT EXISTS experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    site_id TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'complete', 'error')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    variant_a_url TEXT NOT NULL,
    variant_b_url TEXT NOT NULL,
    n_agents INTEGER NOT NULL CHECK (n_agents > 0),
    seed INTEGER NOT NULL,
    provider TEXT NOT NULL CHECK (provider IN ('browserbase', 'local', 'cdp')),
    model TEXT NOT NULL,
    max_cost_usd NUMERIC(10,2) NOT NULL CHECK (max_cost_usd > 0),
    result TEXT,
    lift_abs NUMERIC(10,6),
    lift_rel NUMERIC(10,6),
    p_value NUMERIC(10,6),
    error TEXT,
    idempotency_key TEXT,
    calibration_id TEXT NOT NULL REFERENCES calibrations(calibration_id),
    concurrency INTEGER NOT NULL CHECK (concurrency > 0),
    min_per_arm INTEGER NOT NULL CHECK (min_per_arm > 0),
    min_completion_ratio NUMERIC(3,2) NOT NULL CHECK (min_completion_ratio > 0 AND min_completion_ratio <= 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT UNIQUE NOT NULL,
    experiment_id UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    prompt_id TEXT NOT NULL,
    variant TEXT NOT NULL CHECK (variant IN ('A', 'B')),
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed', 'timeout')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    purchased BOOLEAN NOT NULL DEFAULT FALSE,
    metrics_path TEXT,
    events_jsonb JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS variant_metrics (
    experiment_id UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    variant TEXT NOT NULL CHECK (variant IN ('A', 'B')),
    n INTEGER NOT NULL CHECK (n >= 0),
    purchases INTEGER NOT NULL CHECK (purchases >= 0),
    cr NUMERIC(10,6) NOT NULL CHECK (cr >= 0 AND cr <= 1),
    add_to_cart_rate NUMERIC(10,6),
    begin_checkout_rate NUMERIC(10,6),
    purchase_rate NUMERIC(10,6),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (experiment_id, variant)
);

CREATE TABLE IF NOT EXISTS experiment_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    type TEXT NOT NULL,
    payload_jsonb JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Personas tables (existing)
CREATE TABLE IF NOT EXISTS personas (
    persona_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    payload JSONB NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS persona_prompts (
    persona_id TEXT NOT NULL,
    site_domain TEXT NOT NULL,
    prompt TEXT NOT NULL,
    temperature NUMERIC(3,2) NOT NULL,
    regenerated BOOLEAN NOT NULL DEFAULT FALSE,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (persona_id, site_domain, generated_at)
);

CREATE TABLE IF NOT EXISTS site_vocab (
    site_domain TEXT NOT NULL,
    vocab JSONB NOT NULL,
    scraped_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (site_domain, scraped_at)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_calibrations_user_id ON calibrations(user_id);
CREATE INDEX IF NOT EXISTS idx_calibrations_site_id ON calibrations(site_id);
CREATE INDEX IF NOT EXISTS idx_calibrations_user_site ON calibrations(user_id, site_id);
CREATE INDEX IF NOT EXISTS idx_calibrations_status ON calibrations(status);
CREATE INDEX IF NOT EXISTS idx_calibrations_started_at ON calibrations(started_at);
CREATE INDEX IF NOT EXISTS idx_calibrations_idempotency ON calibrations(idempotency_key) WHERE idempotency_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_ga_snapshots_user_id ON ga_snapshots(user_id);
CREATE INDEX IF NOT EXISTS idx_ga_snapshots_site_id ON ga_snapshots(site_id);
CREATE INDEX IF NOT EXISTS idx_ga_snapshots_user_site ON ga_snapshots(user_id, site_id);
CREATE INDEX IF NOT EXISTS idx_ga_snapshots_taken_at ON ga_snapshots(taken_at);

CREATE INDEX IF NOT EXISTS idx_traffic_distributions_calibration_id ON traffic_distributions(calibration_id);
CREATE INDEX IF NOT EXISTS idx_traffic_distributions_dimension ON traffic_distributions(dimension);

CREATE INDEX IF NOT EXISTS idx_event_rates_calibration_id ON event_rates(calibration_id);
CREATE INDEX IF NOT EXISTS idx_event_rates_event ON event_rates(event);

CREATE INDEX IF NOT EXISTS idx_synthetic_prompts_calibration_id ON synthetic_prompts(calibration_id);
CREATE INDEX IF NOT EXISTS idx_synthetic_prompts_persona_id ON synthetic_prompts(persona_id);

CREATE INDEX IF NOT EXISTS idx_experiments_user_id ON experiments(user_id);
CREATE INDEX IF NOT EXISTS idx_experiments_site_id ON experiments(site_id);
CREATE INDEX IF NOT EXISTS idx_experiments_user_site ON experiments(user_id, site_id);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_started_at ON experiments(started_at);
CREATE INDEX IF NOT EXISTS idx_experiments_calibration_id ON experiments(calibration_id);
CREATE INDEX IF NOT EXISTS idx_experiments_idempotency ON experiments(idempotency_key) WHERE idempotency_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_agent_sessions_session_id ON agent_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_experiment_id ON agent_sessions(experiment_id);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_variant ON agent_sessions(variant);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_status ON agent_sessions(status);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_started_at ON agent_sessions(started_at);

CREATE INDEX IF NOT EXISTS idx_variant_metrics_experiment_id ON variant_metrics(experiment_id);
CREATE INDEX IF NOT EXISTS idx_variant_metrics_variant ON variant_metrics(variant);

CREATE INDEX IF NOT EXISTS idx_experiment_events_experiment_id ON experiment_events(experiment_id);
CREATE INDEX IF NOT EXISTS idx_experiment_events_ts ON experiment_events(ts);
CREATE INDEX IF NOT EXISTS idx_experiment_events_type ON experiment_events(type);

CREATE INDEX IF NOT EXISTS idx_personas_user_id ON personas(user_id);
CREATE INDEX IF NOT EXISTS idx_personas_window_end ON personas(window_end);
CREATE INDEX IF NOT EXISTS idx_personas_generated_at ON personas(generated_at);

CREATE INDEX IF NOT EXISTS idx_persona_prompts_persona_id ON persona_prompts(persona_id);
CREATE INDEX IF NOT EXISTS idx_persona_prompts_site_domain ON persona_prompts(site_domain);
CREATE INDEX IF NOT EXISTS idx_persona_prompts_generated_at ON persona_prompts(generated_at);

CREATE INDEX IF NOT EXISTS idx_site_vocab_site_domain ON site_vocab(site_domain);
CREATE INDEX IF NOT EXISTS idx_site_vocab_scraped_at ON site_vocab(scraped_at);

-- Functions for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updating timestamps
CREATE TRIGGER update_calibrations_updated_at BEFORE UPDATE ON calibrations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_experiments_updated_at BEFORE UPDATE ON experiments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_sessions_updated_at BEFORE UPDATE ON agent_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_variant_metrics_updated_at BEFORE UPDATE ON variant_metrics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
