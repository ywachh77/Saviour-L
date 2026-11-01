"""SAVIOUR Neural AI — Autonomous Neural Trading Intelligence.

This package implements a custom Temporal Transformer with multi-task
prediction heads for Forex market analysis. All weights are initialized
and trained by Saviour itself — no pretrained foundation models.

Modules:
    architecture  — TemporalTransformer with multi-task heads
    dataset       — Candle → training sequence preparation
    trainer       — Autonomous training pipeline (AdamW, early stopping, checkpoints)
    registry      — Model registry, candidate promotion, rollback
    predictor     — Live prediction engine (BUY/SELL/NO TRADE)
    operations    — Background operations daemon
    memory        — Persistent memory (Supabase)
"""
