### voraus-AD baseline floor — episode AUC-ROC (primary) / AUC-PR
_window=100, test_stride=5, train_stride=30, 30000 train windows, dim=2400 (24 ch, cols_limit=24), seed=42, n_test=1174_

| method | AUC-ROC | AUC-PR | shuffled-AUROC(control) |
|---|---|---|---|
| pca_recon | 0.528 | 0.687 | 0.514 |
| iforest | 0.628 | 0.769 | 0.489 |
| lof | 0.749 | 0.853 | 0.515 |
| knn | 0.772 | 0.872 | 0.496 |

#### per-category AUROC (this category vs the 419-episode normal pool)

| category | pca_recon | iforest | lof | knn | n_anom |
|---|---|---|---|---|---|
| AXIS_FRICTION | 0.340 | 0.814 | 0.976 | 0.992 | 144 |
| ENTANGLED | 0.502 | 0.884 | 0.961 | 0.991 | 10 |
| CAN_WEIGHT | 0.392 | 0.631 | 0.803 | 0.874 | 80 |
| COLLISION_FOAM | 0.426 | 0.769 | 0.825 | 0.847 | 72 |
| MISS_CAN | 0.775 | 0.846 | 0.601 | 0.754 | 11 |
| INVALID_POSITION | 0.597 | 0.484 | 0.741 | 0.718 | 12 |
| COLLISION_CABLE | 0.565 | 0.603 | 0.712 | 0.726 | 48 |
| MOTOR_COMMUTATION | 0.650 | 0.642 | 0.719 | 0.708 | 89 |
| WOBBLING_STATION | 0.689 | 0.575 | 0.606 | 0.621 | 37 |
| AXIS_WEIGHT | 0.677 | 0.509 | 0.672 | 0.686 | 156 |
| COLLISION_CARTON | 0.618 | 0.520 | 0.575 | 0.580 | 22 |
| LOSE_CAN | 0.503 | 0.394 | 0.519 | 0.563 | 74 |
