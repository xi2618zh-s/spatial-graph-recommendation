"""M10: PROJECT_HANDOFF_V2.md §3.2 records that lambda=0 makes the combined
Spatial-LightGCN adjacency numerically identical to vanilla LightGCN's, but
flags that this was only ever checked by hand and should be locked into an
automated test before being used as a public claim again. This is that test."""

import numpy as np
import pytest

from src.data.dataset import GowallaData
from src.data.spatial_graph import build_combined_adj
from src.utils.common import ROOT

COORDS_CSV = ROOT / "data" / "processed" / "poi_coords.csv"


@pytest.mark.skipif(not COORDS_CSV.exists(), reason="run scripts/prepare_data.py first")
def test_lambda_zero_equals_vanilla_lightgcn_adjacency():
    data = GowallaData(ROOT / "data" / "gowalla")
    vanilla = data.norm_adj().tocsr()
    combined = build_combined_adj(data, coords_csv=COORDS_CSV, k=10, lam=0.0, max_dist_km=100)

    diff = (vanilla - combined.tocsr())
    max_abs_diff = float(np.abs(diff.data).max()) if diff.nnz else 0.0
    assert max_abs_diff == 0.0, (
        f"lambda=0 combined adjacency diverged from vanilla LightGCN's by {max_abs_diff} -- "
        "this breaks the clean single-knob-ablation claim in docs/01_spatial_lightgcn.md"
    )


@pytest.mark.skipif(not COORDS_CSV.exists(), reason="run scripts/prepare_data.py first")
def test_lambda_zero_equivalence_holds_regardless_of_spatial_params():
    """The spatial term must cancel out entirely at lambda=0 no matter what
    k/sigma/max_dist it was configured with -- if this ever fails while the
    test above passes, the bug is specifically in how lambda scales the
    spatial term, not in the spatial graph construction itself."""
    data = GowallaData(ROOT / "data" / "gowalla")
    vanilla = data.norm_adj().tocsr()
    for k, max_dist, sigma in [(5, 50, None), (20, 200, 1.0), (10, 100, 0.21)]:
        combined = build_combined_adj(
            data, coords_csv=COORDS_CSV, k=k, lam=0.0, max_dist_km=max_dist, sigma_km=sigma
        )
        diff = (vanilla - combined.tocsr())
        max_abs_diff = float(np.abs(diff.data).max()) if diff.nnz else 0.0
        assert max_abs_diff == 0.0, f"k={k}, max_dist={max_dist}, sigma={sigma}: diff={max_abs_diff}"
