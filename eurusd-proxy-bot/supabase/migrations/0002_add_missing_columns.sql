-- FUSION BOT — columns the runtime actually writes that 0001_init.sql omitted.
--
-- Why this file exists: the local JSONL mirror accepts any key you hand it, so
-- the test suite never noticed these were missing. Supabase is stricter —
-- PostgREST answers an unknown column with
--
--     400 Could not find the 'entry_price' column of 'signals' in the schema cache
--
-- and the whole insert is dropped. Three tables would have lost data from the
-- first real cycle:
--
--   signals.entry_price    the sealed entry price (Signal.entry_price in domain.py)
--   learning_events.outcome  the outcome this event was scored against
--   heartbeats.bucket / regime / signal_id
--                          cycle context merged in from CycleReport.to_dict()
--
-- Idempotent: safe to re-run.

alter table public.signals         add column if not exists entry_price double precision;
alter table public.learning_events add column if not exists outcome text;

alter table public.heartbeats      add column if not exists bucket    text;
alter table public.heartbeats      add column if not exists regime    text;
alter table public.heartbeats      add column if not exists signal_id text;

comment on column public.heartbeats.bucket is
  'The minute bucket this heartbeat was written for, e.g. 2026-10-22T09:15Z.';
comment on column public.learning_events.outcome is
  'WIN | LOSS | TIED | UNRESOLVED — the outcome this learning event scored against.';
