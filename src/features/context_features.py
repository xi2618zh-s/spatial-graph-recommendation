"""Context features (§2.4): properties of the request itself.

`query_ts` (the held-out target's own check-in time) stands in for "now" at
serving time — using it is not leakage, it is the one timestamp point-in-time
features are computed *up to*. Everything else here derives only from the
user's own prefix. Verified against prepare_report.json before implementing:
100% of official-train pairs carry a real timestamp (see docs/02_samples_features.md),
so this group did not need to be degraded per the RISK_REGISTER's fallback.
"""

from datetime import datetime, timezone


def context_feature_row(query_ts: int, last_prefix_ts: int) -> dict:
    dt = datetime.fromtimestamp(query_ts, tz=timezone.utc)
    return {
        "ctx_request_hour": dt.hour,
        "ctx_request_dow": dt.weekday(),
        "ctx_days_since_last_checkin": float((query_ts - last_prefix_ts) / 86400.0),
    }
