# Design note: Spatial-LightGCN

## Motivation
Gowalla is a check-in dataset: items are physical POIs. Tobler's first law
("near things are more related") suggests nearby POIs share visitor
populations. Vanilla LightGCN only propagates along user-item interactions,
so long-tail POIs with few check-ins get poor representations. Geographic
item-item edges densify exactly those neighborhoods, injecting location
signal with zero model changes.

## Method
Propagation matrix: `norm(A_bipartite) + lambda * norm(S_geo)` where S_geo is
a symmetrized geographic kNN graph (haversine, BallTree), Gaussian-kernel
weighted `exp(-(d/sigma)^2)` with sigma = median kNN distance by default,
edges > max_dist_km dropped, sym-normalized independently, zero-padded to the
full (users+items) space. Model, loss, training loop: unchanged LightGCN.

## Why this design (interview talking points)
1. **Single-knob ablation**: lambda=0 provably recovers vanilla LightGCN
   (verified: max abs diff of adjacency = 0.0). Any metric delta is
   attributable to spatial edges alone.
2. **Separate normalization** of the two graphs keeps lambda interpretable as
   "relative strength of geographic vs interaction propagation"; joint
   normalization would entangle it with degree shifts.
3. **Gaussian kernel with auto bandwidth** adapts to the dataset's spatial
   density instead of a hand-tuned distance scale.
4. **Expected failure mode to check**: if most check-ins concentrate in dense
   urban areas, k=10 neighbors may all be near-duplicates (same block);
   the k=20 ablation probes sensitivity.

## Planned runs (after LightGCN reproduction is confirmed)
| config | question answered |
|---|---|
| spatial_lightgcn_gowalla.yaml (k10, lam0.3) | main result |
| ablation_spatial_lam0.1 / lam0.5 | sensitivity to spatial strength |
| ablation_spatial_k20 | sensitivity to neighborhood size |
| (existing lightgcn_gowalla.yaml) | lambda=0 control |

Additional analysis after main runs: metric split by item popularity tercile —
hypothesis: spatial edges help long-tail POIs most.
