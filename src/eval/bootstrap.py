"""Bootstrap confidence intervals over per-user metric arrays.

A single point estimate (the corpus mean) hides how much a result could
shift under a different sample of test users. We resample users with
replacement `n_boot` times, take the mean each time, and report the
percentile interval — this is what lets us say a spatial gain is stable
rather than noise, per the project's numeric-discipline rule.
"""

import numpy as np


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 2020) -> dict:
    """Percentile bootstrap CI for the mean of `values`.

    Returns {"mean", "ci_low", "ci_high", "n_users", "n_boot"}.
    """
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(values.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_users": int(n),
        "n_boot": int(n_boot),
    }
