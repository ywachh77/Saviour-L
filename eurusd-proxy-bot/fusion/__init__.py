"""FUSION BOT — unified EUR/USD research signal engine.

Pure standard-library Python. No pip dependencies. Runs on CPython 3.9+
(GitHub Actions, a laptop, any always-on host).

The package is organised as a single-minute pipeline:

    ingest -> data gate -> features -> regime -> strategy council
    -> meta-model (veto/promoted) -> isotonic calibration -> sealed signal

Every stage is importable and testable on its own. Nothing in this package
places a broker order: the output is a research signal with an honest,
cost-charged paper settlement.
"""

__version__ = "2.0.0"

BOUNDARY = (
    "Research and paper-settlement only. No broker orders, no real money, "
    "no guaranteed-accuracy claim."
)
