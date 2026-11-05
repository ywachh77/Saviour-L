-- FUSION BOT — the provenance columns every row already carries in memory.
--
-- Why this file exists: the runtime has always attached "which build wrote
-- this?" to the rows it saves — `system_version` on candles, signals and
-- settlements, `version` on the housekeeping tables, and version/feature/model
-- numbers on the model blobs. The local JSONL mirror keeps any key it is given,
-- so nothing complained; Supabase rejects the whole insert with
--
--     400 Could not find the 'system_version' column of 'signals' in the schema cache
--
-- and the row is dropped silently. signals and settlements are the two tables a
-- post-mortem actually reads back, so losing them costs the history the whole
-- learning loop is built on.
--
-- Every column is nullable with no backfill: an existing install keeps its rows
-- and simply has no provenance recorded for them, which is the truth. Nothing
-- reads these optimistically.
--
-- Idempotent: safe to re-run.

-- Bars, signals and settlements record the build that produced them.
alter table public.candles_m1    add column if not exists system_version text;
alter table public.signals       add column if not exists system_version text;
alter table public.settlements   add column if not exists system_version text;

-- Housekeeping rows record the same thing under a shorter name, which is what
-- store.py has always written (save_key_pool, claim_job, outbox_append,
-- heartbeat, alert).
alter table public.heartbeats    add column if not exists version text;
alter table public.job_runs      add column if not exists version text;
alter table public.outbox        add column if not exists version text;
alter table public.api_keys      add column if not exists version text;
alter table public.alerts        add column if not exists version text;

-- The two model blobs need to say which feature vector and which weights they
-- were built from: a model trained on an older vector must never be loaded
-- against a newer one. Feature-vector changes bump FEATURE_VERSION precisely so
-- this comparison can be made.
alter table public.model_state   add column if not exists version         text;
alter table public.model_state   add column if not exists feature_version integer;
alter table public.model_state   add column if not exists model_version   text;

alter table public.model_versions add column if not exists system_version  text;
alter table public.model_versions add column if not exists feature_version integer;
alter table public.model_versions add column if not exists model_version   text;

comment on column public.signals.system_version is
  'The Fusion build that sealed this signal; lets a post-mortem separate behaviour before and after a deploy.';
comment on column public.candles_m1.system_version is
  'The Fusion build that last wrote this bar (upserts overwrite, so this is the most recent writer).';
comment on column public.model_state.feature_version is
  'FEATURE_VERSION the stored weights were trained on. A mismatch makes the model incompatible rather than silently wrong.';
