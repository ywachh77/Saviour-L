-- FUSION BOT — add OHLCV columns to candles_m1 that 0001_init.sql omitted.
--
-- Why this file exists: 0001_init.sql created candles_m1 with only
-- timestamp and pair columns. The runtime writes open, high, low, close,
-- volume with every candle insert. Supabase PostgREST answers an unknown
-- column with:
--
--     400 Could not find the 'open' column of 'candles_m1' in the schema cache
--
-- and the insert is dropped. The store catches this and falls back to local
-- JSONL, so the cycle still runs — but nothing is persisted to Supabase,
-- and the fetch scripts that read these columns get 400 errors.
--
-- This means the candles_m1 table was effectively empty in Supabase while
-- the local mirror grew. Every workflow that fetches candles from Supabase
-- (fetch-supabase-candles, neural-train-real, real-data-training,
-- fetch-trending-candles) was failing or returning empty data.
--
-- Idempotent: safe to re-run.

alter table public.candles_m1 add column if not exists open   double precision not null default 0;
alter table public.candles_m1 add column if not exists high   double precision not null default 0;
alter table public.candles_m1 add column if not exists low    double precision not null default 0;
alter table public.candles_m1 add column if not exists close  double precision not null default 0;
alter table public.candles_m1 add column if not exists volume double precision not null default 0;

comment on column public.candles_m1.open   is 'Candle open price';
comment on column public.candles_m1.high   is 'Candle high price';
comment on column public.candles_m1.low    is 'Candle low price';
comment on column public.candles_m1.close  is 'Candle close price';
comment on column public.candles_m1.volume is 'Candle volume';
