-- 0005_add_learned_weights_to_council_weights.sql
-- The council's from_dict() reads both "weights" (legacy) and "learned_weights"
-- (newer). The runtime writes learned_weights via to_dict(), but the column
-- was never declared. Live PostgREST would drop the insert with:
--   400 Could not find the 'learned_weights' column of 'council_weights'
-- This migration adds it as jsonb (matching the existing weights column).

alter table public.council_weights
  add column if not exists learned_weights jsonb not null default '{}'::jsonb;
