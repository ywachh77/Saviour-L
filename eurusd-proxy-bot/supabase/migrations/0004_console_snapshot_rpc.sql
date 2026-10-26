-- 0004_console_snapshot_rpc.sql
-- Read-only, zero-secret telemetry & market-data snapshot RPC for the Operator Console.
-- Preserves strict Row Level Security on all 13 underlying tables while allowing
-- the public publishable client key (anon) to read non-secret research state:
-- masked key-pool counters, model/council state, heartbeats, ledger, and M1 candles.

create or replace function public.get_console_snapshot(candle_limit integer default 740)
returns jsonb
language plpgsql
stable
security definer
set search_path = public
as $$
declare
  lim integer := least(greatest(coalesce(candle_limit, 740), 60), 800);
  result jsonb;
begin
  select jsonb_build_object(
    'project_ref', 'idudlcxkihqnvecwqqoj',
    'generated_at', to_char(now() at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"'),
    'api_keys', coalesce((
      select jsonb_agg(jsonb_build_object(
        'id', ak.id,
        'cursor', coalesce(ak.cursor, 0),
        'updated_at', to_char(ak.updated_at at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"'),
        'slots', coalesce((
          select jsonb_agg(jsonb_build_object(
            'index', coalesce((s->>'index')::int, 0),
            'label', coalesce(s->>'label', 'slot'),
            'enabled', coalesce((s->>'enabled')::boolean, true),
            'day_used', coalesce((s->>'day_used')::int, 0),
            'day_limit', coalesce((s->>'day_limit')::int, 800),
            'minute_used', coalesce((s->>'minute_used')::int, 0),
            'minute_limit', coalesce((s->>'minute_limit')::int, 8),
            'total_calls', coalesce((s->>'total_calls')::int, 0),
            'total_failures', coalesce((s->>'total_failures')::int, 0),
            'last_error', coalesce(s->>'disabled_reason', '')
          ))
          from jsonb_array_elements(coalesce(ak.slots, '[]'::jsonb)) as s
        ), '[]'::jsonb)
      ))
      from public.api_keys ak
    ), '[]'::jsonb),
    'model_state', coalesce((
      select jsonb_agg(jsonb_build_object(
        'id', ms.id,
        'model', ms.model,
        'ensemble', ms.ensemble,
        'calibration_points', ms.calibration_points,
        'mistakes', ms.mistakes,
        'updated_at', to_char(ms.updated_at at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"'),
        'runtime', jsonb_build_object(
          'paused', coalesce((ms.runtime->>'paused')::boolean, false),
          'shadow_recovery', coalesce(ms.runtime->'shadow_recovery', '{}'::jsonb),
          'telegram_format', coalesce(ms.runtime->'telegram_format', '{}'::jsonb),
          'stale_feed_minutes', coalesce((ms.runtime->>'stale_feed_minutes')::int, 0),
          'telegram_offset', coalesce((ms.runtime->>'telegram_offset')::bigint, 0)
        )
      ))
      from public.model_state ms
    ), '[]'::jsonb),
    'council_weights', coalesce((
      select jsonb_agg(jsonb_build_object(
        'id', cw.id,
        'net', cw.net,
        'seen', cw.seen,
        'weights', cw.weights,
        'version', cw.version
      ))
      from public.council_weights cw
    ), '[]'::jsonb),
    'heartbeats', coalesce((
      select jsonb_agg(row_to_json(hb_sub)::jsonb)
      from (
        select
          id, runner, phase, paused, cycles, seal_state, reason,
          to_char(created_at at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as created_at,
          bucket, regime, signal_id
        from (
          select * from public.heartbeats order by created_at desc limit 20
        ) recent_hb
        order by created_at asc
      ) hb_sub
    ), '[]'::jsonb),
    'signals', coalesce((
      select jsonb_agg(row_to_json(sig_sub)::jsonb)
      from (
        select
          signal_id, pair, side,
          to_char(entry_time at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as entry_time,
          to_char(expiry_time at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as expiry_time,
          to_char(generated_at at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as generated_at,
          horizon_minutes, strategy, reason, source,
          probability_up, probability_raw, calibration_state, uncertainty,
          regime, council,
          case when coalesce(settled, false) then '{}'::jsonb else coalesce(features, '{}'::jsonb) end as features,
          spread_pips,
          seal_state, delivery_state, message_id, settled, bucket, entry_price
        from public.signals
        order by entry_time desc
        limit 50
      ) sig_sub
    ), '[]'::jsonb),
    'settlements', coalesce((
      select jsonb_agg(row_to_json(stl_sub)::jsonb)
      from (
        select
          signal_id, outcome, entry_price, exit_price, gross_pips,
          move_pips, net_pips, spread_pips,
          to_char(settled_at at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as settled_at,
          source
        from public.settlements
        order by settled_at desc
        limit 500
      ) stl_sub
    ), '[]'::jsonb),
    'learning_events', coalesce((
      select jsonb_agg(row_to_json(le_sub)::jsonb)
      from (
        select
          id, kind, signal_id, applied, reason, predicted, raw,
          label, squared_error,
          to_char(created_at at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as created_at,
          outcome
        from public.learning_events
        order by created_at desc
        limit 40
      ) le_sub
    ), '[]'::jsonb),
    'candles_m1', coalesce((
      select jsonb_agg(row_to_json(c_all)::jsonb)
      from (
        select
          pair,
          to_char("timestamp" at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as "timestamp",
          "open"::double precision as "open",
          high::double precision as high,
          low::double precision as low,
          "close"::double precision as "close",
          volume::double precision as volume
        from (
          (select * from public.candles_m1 where pair = 'EUR/USD' order by "timestamp" desc limit lim)
          union all
          (select * from public.candles_m1 where pair = 'BTC/USD' order by "timestamp" desc limit lim)
          union all
          (select * from public.candles_m1 where pair = 'ETH/USD' order by "timestamp" desc limit lim)
        ) per_pair
        order by pair asc, "timestamp" asc
      ) c_all
    ), '[]'::jsonb)
  ) into result;

  return result;
end;
$$;

revoke all on function public.get_console_snapshot(integer) from public;
grant execute on function public.get_console_snapshot(integer) to anon, authenticated, service_role;
