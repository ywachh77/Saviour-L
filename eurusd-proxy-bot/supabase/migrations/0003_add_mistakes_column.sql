-- Post-mortem memory: what the bot has already paid to learn.
--
-- One column on the single model_state row. It holds per-setup win/loss counts
-- (regime x session x side), so a pattern the bot has demonstrably lost money
-- on can be refused instead of re-tried. It can only ever block a signal, never
-- create one.
--
-- Nullable and unbackfilled: an existing install simply starts with an empty
-- memory, which is the same state a fresh one starts in. Nothing reads it
-- optimistically -- MistakeMemory.from_dict(None) is an empty memory.

alter table public.model_state
  add column if not exists mistakes jsonb;
