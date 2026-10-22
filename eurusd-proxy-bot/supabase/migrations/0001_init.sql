-- FUSION BOT — initial schema (13 tables, row-level security on every one).
--
-- Run this once in Supabase → SQL Editor. It is idempotent: safe to re-run.
--
-- Security model: RLS is enabled on all 13 tables with NO policies for the
-- anon or authenticated roles. That means the publishable (anon) key can read
-- and write nothing. Only the service_role key — which never leaves your
-- .env and GitHub secrets — can touch this data. Add a policy later only if
-- you deliberately want a dashboard to read it.

create extension if not exists pgcrypto;

-- 1. One-minute candles ------------------------------------------------------
create table if not exists public.candles_m1 (
  timestamp   timestamptz not null,
  pair        text        not null,
  open        numeric     not null,
  high        numeric     not null,
  low         numeric     not null,
  close       numeric     not null,
  volume      numeric     not null default 0,
  primary key (timestamp, pair)
);
create index if not exists candles_m1_pair_ts_idx on public.candles_m1 (pair, timestamp desc);

-- 2. Sealed signals ----------------------------------------------------------
create table if not exists public.signals (
  signal_id         text primary key,
  pair              text not null,
  side              text not null,
  entry_time        timestamptz not null,
  expiry_time       timestamptz not null,
  generated_at      timestamptz not null,
  horizon_minutes   integer not null default 5,
  strategy          text not null,
  reason            text not null,
  source            text not null,
  probability_up    double precision,
  probability_raw   double precision,
  calibration_state text not null default 'RAW',
  uncertainty       double precision,
  regime            text,
  council           jsonb not null default '[]'::jsonb,
  features          jsonb not null default '{}'::jsonb,
  spread_pips       double precision not null default 1.2,
  seal_state        text not null default 'SIGNAL',
  delivery_state    text not null default 'PENDING',
  message_id        text,
  settled           boolean not null default false,
  bucket            text
);
create index if not exists signals_entry_idx on public.signals (entry_time desc);
create index if not exists signals_pending_idx on public.signals (pair, settled) where settled = false;

-- 3. Cost-charged settlements ------------------------------------------------
create table if not exists public.settlements (
  signal_id   text primary key,
  outcome     text not null,           -- WIN | LOSS | TIED | UNRESOLVED
  entry_price double precision,
  exit_price  double precision,
  gross_pips  double precision,
  move_pips   double precision,
  net_pips    double precision,        -- NULL while UNRESOLVED, never guessed
  spread_pips double precision not null,
  settled_at  timestamptz not null,
  source      text not null
);
create index if not exists settlements_settled_idx on public.settlements (settled_at desc);

-- 4. Champion model + runtime state (single row) -----------------------------
create table if not exists public.model_state (
  id                 text primary key,  -- 'champion'
  model              jsonb,
  ensemble           jsonb,
  runtime            jsonb,
  calibration_points jsonb,
  updated_at         timestamptz not null default now()
);

-- 5. Every nightly candidate and its gate report -----------------------------
create table if not exists public.model_versions (
  version_id text primary key,
  report     jsonb not null,
  created_at timestamptz not null default now()
);

-- 6. Learned council weights -------------------------------------------------
create table if not exists public.council_weights (
  id      text primary key,            -- 'council'
  net     jsonb not null default '{}'::jsonb,
  seen    jsonb not null default '{}'::jsonb,
  weights jsonb not null default '{}'::jsonb,
  version integer not null default 1
);

-- 7. Learning audit trail ----------------------------------------------------
create table if not exists public.learning_events (
  id           bigserial primary key,
  kind         text not null,
  signal_id    text,
  applied      boolean not null default false,
  reason       text,
  predicted    double precision,
  raw          double precision,
  label        integer,
  squared_error double precision,
  created_at   timestamptz not null default now()
);
create index if not exists learning_events_created_idx on public.learning_events (created_at desc);

-- 8. Job slots (idempotency) -------------------------------------------------
create table if not exists public.job_runs (
  id         bigserial primary key,
  job        text not null,
  bucket     text not null,
  runner     text not null,
  created_at timestamptz not null default now(),
  unique (job, bucket)                 -- the second writer loses, by design
);

-- 9. Heartbeats (the only liveness signal) -----------------------------------
create table if not exists public.heartbeats (
  id         bigserial primary key,
  runner     text not null,
  phase      text not null,
  paused     boolean,
  cycles     integer,
  seal_state text,
  reason     text,
  created_at timestamptz not null default now()
);
create index if not exists heartbeats_runner_created_idx on public.heartbeats (runner, created_at desc);

-- 10. API key pool metadata (NEVER the key itself) ---------------------------
create table if not exists public.api_keys (
  id         text primary key,         -- 'keypool'
  slots      jsonb not null default '[]'::jsonb,
  cursor     integer not null default 0,
  updated_at timestamptz not null default now()
);

-- 11. Per-key usage counters -------------------------------------------------
create table if not exists public.key_usage (
  id          bigserial primary key,
  key_index   integer not null,
  label       text not null,
  day_bucket  text not null,
  minute_used integer not null default 0,
  day_used    integer not null default 0,
  created_at  timestamptz not null default now()
);

-- 12. Outbox (every message and its true delivery state) ---------------------
create table if not exists public.outbox (
  message_key text primary key,
  text        text not null,
  kind        text not null default 'signal',
  attempts    integer not null default 0,
  state       text not null default 'PENDING',  -- PENDING|SENT|FAILED|UNRESOLVED|SUPPRESSED
  message_id  text,
  last_error  text,
  created_at  timestamptz not null default now()
);

-- 13. Alerts that need a human -----------------------------------------------
create table if not exists public.alerts (
  id         bigserial primary key,
  kind       text not null,
  message    text not null,
  dedupe_key text,
  created_at timestamptz not null default now()
);
create index if not exists alerts_created_idx on public.alerts (created_at desc);

-- Row-level security on every table ------------------------------------------
alter table public.candles_m1      enable row level security;
alter table public.signals         enable row level security;
alter table public.settlements     enable row level security;
alter table public.model_state     enable row level security;
alter table public.model_versions  enable row level security;
alter table public.council_weights enable row level security;
alter table public.learning_events enable row level security;
alter table public.job_runs        enable row level security;
alter table public.heartbeats      enable row level security;
alter table public.api_keys        enable row level security;
alter table public.key_usage       enable row level security;
alter table public.outbox          enable row level security;
alter table public.alerts          enable row level security;

comment on table public.api_keys is
  'Key pool metadata only. Plaintext provider keys are never stored here or anywhere else.';
comment on table public.settlements is
  'net_pips is NULL while an outcome is UNRESOLVED. An unresolved outcome is never converted into a win or a loss.';
comment on table public.outbox is
  'state is SENT only after Telegram confirms a message id. An unknown result is recorded as UNRESOLVED and raises an alert.';
