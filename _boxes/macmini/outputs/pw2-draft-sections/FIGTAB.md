# Figure / table index, after (both drafts), with the before number and page

## DRAFT-v2

| kind | number | label | caption lead | page | section | before: number, page |
|---|---|---|---|---|---|---|
| figure | 1 | `fig:task-shapes` | The Shapes3D task. | 5 | 3 Experiments | 1, p. 4 |
| figure | 2 | `fig:task-cmazet` | The CMAZE-T task. | 6 | 3 Experiments | 2, p. 5 |
| figure | 3 | `fig:shapes-per-model` | Each model succeeding and failing on held-out combinations (Shapes3D, seed 0, K* | 10 | 4.1 Composition on Shapes3D | 3, p. 8 |
| figure | 4 | `fig:shapes-baseline-wins` | Held-out requests on which a baseline succeeds and the n-CLU fails (seed 0). | 11 | 4.1 Composition on Shapes3D | 4, p. 9 |
| figure | 5 | `fig:ladder` | The ladder at K* against data-matched baselines, 3 seeds. | 13 | 4.2 The n-way ladder | 5, p. 11 |
| figure | 6 | `fig:crossover` | The K-curve, validation, 3 seeds at every K. | 28 | D The K-curve: composition and the memory footprint against K | 6, p. 27 |
| figure | 7 | `fig:deletion` | Deletion on the single design at K* , rows 3-4, test, fit pool 131,072 rows, see | 30 | F The earlier composition cell: byte and parameter match, and the ceiling ladder | 7, p. 29 |
| figure | 8 | `fig:adapt` | Encoder adaptation margins. | 32 | H Light encoder adaptation in full | 8, p. 31 |
| figure | 9 | `fig:maze-wm` | CMAZE-T: the n-CLU against the best matched baseline under four input choices, a | 37 | N Maze world-model rows | 9, p. 36 |
| figure | 10 | `fig:cmazet-families` | Held-out CMAZE-T transitions, one per rule family, with every model's next state | 39 | N Maze world-model rows | 10, p. 38 |
| figure | 11 | `fig:cmazet-per-arm` | Each CMAZE-T model succeeding and failing (words plus observation, test, seed 0) | 40 | N Maze world-model rows | 11, p. 39 |
| figure | 12 | `fig:cmazet-rel-per-arm` | Each CMAZE-T model succeeding and failing with relational coding (words plus obs | 41 | N Maze world-model rows | 12, p. 40 |
| figure | 13 | `fig:cmazet-door-tint` | Door-then-tint transitions: the move opens a door with the key in the pocket and | 42 | N Maze world-model rows | 13, p. 41 |
| table | 1 | `tab:t7` | Composition at K* =32 against data-matched baselines, Shapes3D value hold-out, T | 8 | 4.1 Composition on Shapes3D | 1, p. 7 |
| table | 2 | `tab:ledger-kstar` | The footprint breakdown of the K* =32 object, per seed, never netted. | 9 | 4.1 Composition on Shapes3D | 2, p. 9 |
| table | 3 | `tab:ladder` | The ladder at K* =32, n=3 seeds, distinct valid novel tuples per combination, pe | 12 | 4.2 The n-way ladder | 3, p. 10 |
| table | 4 | `tab:del` | Deletion on the single design, the four measurements, K* =32, rows 3-4, TEST, fi | 14 | 4.3 Deletion on Shapes3D | 4, p. 12 |
| table | 5 | `tab:hole` | Refusal DISPLAY rows. A display is never a result. | 14 | 4.5 Refusal through the hole | 5, p. 14 |
| table | 6 | `tab:gentable-kstar` | The generation table, K* single design, one declared evaluation run, generation- | 26 | B Tab. [tab:t7]: what it does and does not claim | 6, p. 25 |
| table | 7 | `tab:lace-poe` | LACE and PoE at their own settings, Shapes3D 3dshapes18,kim_disentangling_2019 v | 26 | B Tab. [tab:t7]: what it does and does not claim | 7, p. 25 |
| table | 8 | `tab:k5` | Ladder at K=5, the sweep's endpoint, rebuilt n-CLU, 3 seeds. | 27 | C The K=5 endpoint of the sweep | 8, p. 26 |
| table | 9 | `tab:kcurve` | The K-curve, validation, symmetric-triangle convention. | 27 | D The K-curve: composition and the memory footprint against K | 9, p. 26 |
| table | 10 | `tab:kcurve3` | The K-curve at n=3 seeds on every row, validation (a smoke, not a claim). | 28 | D The K-curve: composition and the memory footprint against K | 10, p. 27 |
| table | 11 | `tab:impl` | Deletion on Shapes3D, matched on the n-CLU, implementations 1 and 2, bands not s | 29 | E Deletion on Shapes3D: two implementations, and the failure they agree on | 11, p. 28 |
| table | 12 | `tab:ablations` | The two ablations, earlier composition cell, TEST, n=3 seeds, one pin (524,288-b | 29 | F The earlier composition cell: byte and parameter match, and the ceiling ladder | 12, p. 28 |
| table | 13 | `tab:rivals-del` | Deletion cost for the data-matched baselines: retrain without the rows, on the n | 30 | G Deletion comparators in full (fit pool, 0 test runs) | 13, p. 29 |
| table | 14 | `tab:rivals-del-seeds` | The n-CLU after deletion against each baseline retrained without the same rows,  | 31 | G Deletion comparators in full (fit pool, 0 test runs) | 14, p. 30 |
| table | 15 | `tab:adapt` | Encoder adaptation at K* , TEST, 3 seeds x 3 configurations (nine arm-runs, thre | 31 | H Light encoder adaptation in full | 15, p. 30 |
| table | 16 | `tab:encloop` | Encoder loop, shortened decoder (400 joint steps, 8,192 images), 3 seeds, one de | 32 | I The encoder loop in full (shortened-decoder instrument, and the longer leg) | 16, p. 31 |
| table | 17 | `tab:bard-gpu` | The unfreezing decomposition on the longer leg only (4,000 joint steps, 32,768 t | 32 | I The encoder loop in full (shortened-decoder instrument, and the longer leg) | 17, p. 31 |
| table | 18 | `tab:crr` | Declared criteria of the retention law on the field-mode n-CLU, 49 held-out comb | 33 | K The retention law on the field-mode n-CLU in full | 18, p. 32 |
| table | 19 | `tab:cmazet` | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seed | 38 | N Maze world-model rows | 19, p. 37 |
| table | 19 | `tab:cmaze-del` | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seed | 38 | N Maze world-model rows | 19, p. 37 |
| table | 20 | `tab:cmazet-sym` | CMAZE-T with symmetric inputs: every condition, unseen n-way, the engine's judge | 43 | N Maze world-model rows | 20, p. 42 |

numbers changed before -> after: 0; labels before 34, after 34, same set: True

## DRAFT-v2-style

| kind | number | label | caption lead | page | section | before: number, page |
|---|---|---|---|---|---|---|
| figure | 1 | `fig:task-shapes` | The Shapes3D task. | 6 | 3 Experiments | 1, p. 4 |
| figure | 2 | `fig:task-cmazet` | The CMAZE-T task. | 7 | 3 Experiments | 2, p. 5 |
| figure | 3 | `fig:shapes-per-model` | Each model succeeding and failing on held-out combinations (Shapes3D, seed 0, K* | 9 | 4.1 Composition on Shapes3D | 3, p. 9 |
| figure | 4 | `fig:shapes-baseline-wins` | Held-out requests on which a baseline succeeds and the n-CLU fails (seed 0). | 10 | 4.1 Composition on Shapes3D | 4, p. 10 |
| figure | 5 | `fig:ladder` | The ladder at K* against data-matched baselines, 3 seeds. | 13 | 4.2 The n-way ladder | 5, p. 12 |
| figure | 6 | `fig:crossover` | The K-curve, validation, 3 seeds at every K. | 28 | D The K-curve: composition and the memory footprint against K | 6, p. 28 |
| figure | 7 | `fig:deletion` | Deletion on the single design at K* , rows 3-4, test, fit pool 131,072 rows, see | 30 | F The earlier composition cell: byte and parameter match, and the ceiling ladder | 7, p. 30 |
| figure | 8 | `fig:adapt` | Encoder adaptation margins. | 32 | H Light encoder adaptation in full | 8, p. 32 |
| figure | 9 | `fig:maze-wm` | CMAZE-T: the n-CLU against the best matched baseline under four input choices, a | 37 | N Maze world-model rows | 9, p. 37 |
| figure | 10 | `fig:cmazet-families` | Held-out CMAZE-T transitions, one per rule family, with every model's next state | 39 | N Maze world-model rows | 10, p. 39 |
| figure | 11 | `fig:cmazet-per-arm` | Each CMAZE-T model succeeding and failing (words plus observation, test, seed 0) | 40 | N Maze world-model rows | 11, p. 40 |
| figure | 12 | `fig:cmazet-rel-per-arm` | Each CMAZE-T model succeeding and failing with relational coding (words plus obs | 41 | N Maze world-model rows | 12, p. 41 |
| figure | 13 | `fig:cmazet-door-tint` | Door-then-tint transitions: the move opens a door with the key in the pocket and | 42 | N Maze world-model rows | 13, p. 42 |
| table | 1 | `tab:t7` | Composition at K* =32 against data-matched baselines, Shapes3D value hold-out, T | 10 | 4.1 Composition on Shapes3D | 1, p. 7 |
| table | 2 | `tab:ledger-kstar` | The footprint breakdown of the K* =32 object, per seed, never netted. | 11 | 4.1 Composition on Shapes3D | 2, p. 8 |
| table | 3 | `tab:ladder` | The ladder at K* =32, n=3 seeds, distinct valid novel tuples per combination, pe | 11 | 4.2 The n-way ladder | 3, p. 11 |
| table | 4 | `tab:del` | Deletion on the single design, the four measurements, K* =32, rows 3-4, TEST, fi | 14 | 4.3 Deletion on Shapes3D | 4, p. 13 |
| table | 5 | `tab:hole` | Refusal DISPLAY rows. A display is never a result. | 15 | 4.5 Refusal through the hole | 5, p. 14 |
| table | 6 | `tab:gentable-kstar` | The generation table, K* single design, one declared evaluation run, generation- | 26 | B Tab. [tab:t7]: what it does and does not claim | 6, p. 26 |
| table | 7 | `tab:lace-poe` | LACE and PoE at their own settings, Shapes3D 3dshapes18,kim_disentangling_2019 v | 26 | B Tab. [tab:t7]: what it does and does not claim | 7, p. 26 |
| table | 8 | `tab:k5` | Ladder at K=5, the sweep's endpoint, rebuilt n-CLU, 3 seeds. | 27 | C The K=5 endpoint of the sweep | 8, p. 27 |
| table | 9 | `tab:kcurve` | The K-curve, validation, symmetric-triangle convention. | 27 | D The K-curve: composition and the memory footprint against K | 9, p. 27 |
| table | 10 | `tab:kcurve3` | The K-curve at n=3 seeds on every row, validation (a smoke, not a claim). | 28 | D The K-curve: composition and the memory footprint against K | 10, p. 28 |
| table | 11 | `tab:impl` | Deletion on Shapes3D, matched on the n-CLU, implementations 1 and 2, bands not s | 29 | E Deletion on Shapes3D: two implementations, and the failure they agree on | 11, p. 29 |
| table | 12 | `tab:ablations` | The two ablations, earlier composition cell, TEST, n=3 seeds, one pin (524,288-b | 30 | F The earlier composition cell: byte and parameter match, and the ceiling ladder | 12, p. 30 |
| table | 13 | `tab:rivals-del` | Deletion cost for the data-matched baselines: retrain without the rows, on the n | 31 | G Deletion comparators in full (fit pool, 0 test runs) | 13, p. 31 |
| table | 14 | `tab:rivals-del-seeds` | The n-CLU after deletion against each baseline retrained without the same rows,  | 31 | G Deletion comparators in full (fit pool, 0 test runs) | 14, p. 31 |
| table | 15 | `tab:adapt` | Encoder adaptation at K* , TEST, 3 seeds x 3 configurations (nine arm-runs, thre | 32 | H Light encoder adaptation in full | 15, p. 32 |
| table | 16 | `tab:encloop` | Encoder loop, shortened decoder (400 joint steps, 8,192 images), 3 seeds, one de | 33 | I The encoder loop in full (shortened-decoder instrument, and the longer leg) | 16, p. 33 |
| table | 17 | `tab:bard-gpu` | The unfreezing decomposition on the longer leg only (4,000 joint steps, 32,768 t | 33 | I The encoder loop in full (shortened-decoder instrument, and the longer leg) | 17, p. 33 |
| table | 18 | `tab:crr` | Declared criteria of the retention law on the field-mode n-CLU, 49 held-out comb | 34 | K The retention law on the field-mode n-CLU in full | 18, p. 34 |
| table | 19 | `tab:cmazet` | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seed | 38 | N Maze world-model rows | 19, p. 38 |
| table | 19 | `tab:cmaze-del` | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seed | 38 | N Maze world-model rows | 19, p. 38 |
| table | 20 | `tab:cmazet-sym` | CMAZE-T with symmetric inputs: every condition, unseen n-way, the engine's judge | 43 | N Maze world-model rows | 20, p. 43 |

numbers changed before -> after: 0; labels before 34, after 34, same set: True

