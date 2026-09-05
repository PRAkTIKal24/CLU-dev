| route | arm | family | metric | dividend ± SE | 2SE? | +0 B margin | admissible | live | anchor | bytes | grade |
|---|---|---|---|---|---|---|---|---|---|---|---|
| route1 | endpoint_write | aggregate | neg_mae | n/a | no | +nan | 0/3 (0%) | no | yes | 54.56x (unmatched) | **abstain** |
| route1 | endpoint_write | manifold | r2 | n/a | no | +nan | 0/3 (0%) | no | yes | 52.00x (unmatched) | **abstain** |
| route1 | endpoint_write | overload | decode | -0.0278 ± 0.0139 | no | +0.2639 | 3/3 (100%) | no | yes | 478.20x (unmatched) | **le_zero_vote** |
| route1 | path_write@0.03 | overload | decode | -0.2639 ± 0.0139 | no | +0.0278 | 3/3 (100%) | no | yes | 478.20x (unmatched) | **le_zero_vote** |
| route1 | path_write@0.3 | aggregate | neg_mae | n/a | no | +nan | 0/3 (0%) | no | yes | 54.56x (unmatched) | **abstain** |
| route1 | path_write@0.3 | manifold | r2 | n/a | no | +nan | 0/3 (0%) | no | yes | 52.00x (unmatched) | **abstain** |
| route1 | path_write@0.3 | overload | decode | -0.2917 ± 0.0636 | no | +0.0000 | 3/3 (100%) | no | yes | 478.20x (unmatched) | **le_zero_vote** |
| route1 | path_write@3 | overload | decode | -0.7083 (1 seed) | no | -0.2917 | 1/3 (33%) | no | yes | 478.20x (unmatched) | **le_zero_vote** |
| route1 | path_write@30 | overload | decode | n/a | no | +nan | 0/3 (0%) | no | yes | 478.20x (unmatched) | **abstain** |
| route1 | traj+path@0.3 | aggregate | neg_mae | n/a | no | +nan | 0/3 (0%) | no | yes | 54.56x (unmatched) | **abstain** |
| route1 | traj+path@0.3 | manifold | r2 | n/a | no | +nan | 0/3 (0%) | no | yes | 52.00x (unmatched) | **abstain** |
| route1 | traj+path@0.3 | overload | decode | -0.4722 ± 0.0501 | no | -0.1806 | 3/3 (100%) | no | yes | 478.20x (unmatched) | **le_zero_vote** |
| route1 | traj_write@0.03 | overload | decode | -0.1111 ± 0.0278 | no | +0.1806 | 3/3 (100%) | no | yes | 478.20x (unmatched) | **le_zero_vote** |
| route1 | traj_write@0.3 | aggregate | neg_mae | n/a | no | +nan | 0/3 (0%) | no | yes | 54.56x (unmatched) | **abstain** |
| route1 | traj_write@0.3 | manifold | r2 | n/a | no | +nan | 0/3 (0%) | no | yes | 52.00x (unmatched) | **abstain** |
| route1 | traj_write@0.3 | overload | decode | -0.1806 ± 0.0139 | no | +0.1111 | 3/3 (100%) | no | yes | 478.20x (unmatched) | **le_zero_vote** |
| route1 | traj_write@3 | overload | decode | -0.1944 ± 0.0278 | no | +0.0972 | 3/3 (100%) | no | yes | 478.20x (unmatched) | **le_zero_vote** |
| route1 | traj_write@30 | overload | decode | -0.2639 ± 0.0773 | no | +0.0278 | 3/3 (100%) | no | yes | 478.20x (unmatched) | **le_zero_vote** |

**Excluded cells (every one, with its reason — silent filtering is forbidden):**

| route/arm/family | seed | reason |
|---|---|---|
| route1/endpoint_write/aggregate | 0 | endpoint write loss 0.2463 > 0.05 |
| route1/endpoint_write/aggregate | 1 | endpoint write loss 0.3612 > 0.05 |
| route1/endpoint_write/aggregate | 2 | endpoint write loss 0.2862 > 0.05 |
| route1/endpoint_write/manifold | 0 | endpoint write loss 0.2494 > 0.05 |
| route1/endpoint_write/manifold | 1 | endpoint write loss 0.3808 > 0.05 |
| route1/endpoint_write/manifold | 2 | endpoint write loss 0.2523 > 0.05 |
| route1/path_write@0.3/aggregate | 0 | endpoint write loss 0.2444 > 0.05 |
| route1/path_write@0.3/aggregate | 1 | endpoint write loss 0.4330 > 0.05 |
| route1/path_write@0.3/aggregate | 2 | endpoint write loss 0.3208 > 0.05 |
| route1/path_write@0.3/manifold | 0 | endpoint write loss 0.2494 > 0.05 |
| route1/path_write@0.3/manifold | 1 | endpoint write loss 0.2861 > 0.05 |
| route1/path_write@0.3/manifold | 2 | endpoint write loss 0.2523 > 0.05 |
| route1/path_write@3/overload | 0 | lambda_min<0 (-0.7197) |
| route1/path_write@3/overload | 2 | lambda_min<0 (-0.4793) |
| route1/path_write@30/overload | 0 | endpoint write loss 0.0598 > 0.05 |
| route1/path_write@30/overload | 1 | lambda_min<0 (-1.3022) |
| route1/path_write@30/overload | 2 | endpoint write loss 0.0589 > 0.05 |
| route1/traj+path@0.3/aggregate | 0 | endpoint write loss 0.2444 > 0.05 |
| route1/traj+path@0.3/aggregate | 1 | endpoint write loss 0.7821 > 0.05 |
| route1/traj+path@0.3/aggregate | 2 | endpoint write loss 0.3927 > 0.05 |
| route1/traj+path@0.3/manifold | 0 | endpoint write loss 0.2494 > 0.05 |
| route1/traj+path@0.3/manifold | 1 | endpoint write loss 0.6023 > 0.05 |
| route1/traj+path@0.3/manifold | 2 | endpoint write loss 0.3307 > 0.05 |
| route1/traj_write@0.3/aggregate | 0 | endpoint write loss 0.2463 > 0.05 |
| route1/traj_write@0.3/aggregate | 1 | endpoint write loss 0.5033 > 0.05 |
| route1/traj_write@0.3/aggregate | 2 | endpoint write loss 0.3441 > 0.05 |
| route1/traj_write@0.3/manifold | 0 | endpoint write loss 0.2494 > 0.05 |
| route1/traj_write@0.3/manifold | 1 | endpoint write loss 0.6363 > 0.05 |
| route1/traj_write@0.3/manifold | 2 | endpoint write loss 0.2877 > 0.05 |