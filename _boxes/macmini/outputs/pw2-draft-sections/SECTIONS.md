# Section index and page counts, before -> after (PW-2 item 2)

From exact heading markers at shipout (`page_markers_c.py`, marker builds with every `\newlabel` identical to the unmarked build). "len" = text-flow length in pages from the heading to the next heading of the same or higher level (floats count where they land). Main text = Introduction heading to the References heading.

## DRAFT-v2, before: 42 pages; main text (Introduction -> References) 17.40 pages (Introduction on p. 2, References on p. 19)

| # | heading | page | len (pages) |
|---|---|---|---|
| 1 | Introduction: a naive unit, and why its three properties matter | 2 | 2.74 |
| 1.1 | &nbsp;&nbsp;Positioning: a naive unit, compared with naive units | 2 | 1.01 |
| 1.2 | &nbsp;&nbsp;Why the three properties matter: world models need all three at once | 3 | 0.28 |
| 1.3 | &nbsp;&nbsp;What we show, and at what thickness | 3 | 1.40 |
| 2 | The object, the judge and the memory footprint | 4 | 1.90 |
| 3 | The n-CLU on Shapes3D: the single design at $\Kstar$ | 6 | 8.69 |
| 3.1 | &nbsp;&nbsp;Composition at $\Kstar$: rows 3 and 4, three scored seeds | 6 | 2.93 |
| 3.2 | &nbsp;&nbsp;The $n$-way ladder at $\Kstar$, with its inference note | 9 | 2.69 |
| 3.3 | &nbsp;&nbsp;Deletion on the single design: value, conjunction, item | 12 | 0.91 |
| 3.4 | &nbsp;&nbsp;What the baselines can do instead: retrain without, or approximate | 13 | 0.33 |
| 3.5 | &nbsp;&nbsp;Refusal after deletion: the hole, a separate mechanism | 13 | 0.62 |
| 3.6 | &nbsp;&nbsp;Two ablations, each keeping one property | 14 | 0.45 |
| 3.7 | &nbsp;&nbsp;The encoder: frozen by design, lightly adapted; the encoder loop licenses no n-CLU claim | 14 | 0.47 |
| 4 | The same properties on a maze | 15 | 1.67 |
| 5 | Limitations of the naive unit, and a leaving note | 17 | 1.72 |
| 5.1 | &nbsp;&nbsp;What the naive unit does not do (measured, not hedged) | 17 | 0.75 |
| 5.2 | &nbsp;&nbsp;The clue: the n-CLU is the zero-dynamics limit of a settling unit | 17 | 0.29 |
| 5.3 | &nbsp;&nbsp;The theory we carry: the Latent Dynamics Model class | 18 | 0.42 |
| 5.4 | &nbsp;&nbsp;Leaving note | 18 | 0.23 |
| 6 | Follow-up: a full evaluation of the physics CLU | 18 | 0.22 |
|  | &nbsp;&nbsp;AI use statement | 18 | 0.12 |
|  | &nbsp;&nbsp;Ethics statement | 19 | 0.15 |
|  | &nbsp;&nbsp;Reproducibility statement | 19 | 0.19 |
|  | References | 19 | 4.60 |
|  | Appendices | 24 | 0.12 |
| A | Reporting conventions | 24 | 0.41 |
| B | Tab.~\ref{tab:t7}: what it does and does not claim | 24 | 1.13 |
| C | The $K=5$ endpoint of the sweep | 25 | 0.58 |
| D | The $K$-curve: composition and the memory footprint against $K$ | 26 | 1.59 |
| E | Deletion on Shapes3D: two implementations, and the failure they agree on | 27 | 0.96 |
| F | The earlier composition cell: byte and parameter match, and the ceiling ladder | 28 | 0.95 |
| G | Deletion comparators in full (fit pool, 0 test runs) | 29 | 0.66 |
| H | Light encoder adaptation in full | 30 | 0.44 |
| I | The encoder loop in full (shortened-decoder instrument, and the longer leg) | 30 | 1.16 |
| J | The zero-dynamics identity: conditions and provenance | 32 | 0.73 |
| K | The retention law on the field-mode n-CLU in full | 32 | 0.85 |
| L | Prior derivations | 33 | 1.31 |
| M | What an evaluation must contain: the ten requirements | 34 | 0.48 |
| N | Maze world-model rows | 35 | 1.40 |
| O | The encoder: adaptation and the unfreezing decomposition | 36 | 2.11 |
| P | The two laws on this study's n-CLU, stated in full | 38 | 1.86 |
| Q | Further limitations | 40 | 0.82 |
| R | Placeholder index | 41 | 0.29 |
| S | Declared not-runs | 41 | 0.91 |

## DRAFT-v2, after: 43 pages; main text (Introduction -> References) 17.67 pages (Introduction on p. 2, References on p. 19)

| # | heading | page | len (pages) |
|---|---|---|---|
| 1 | Introduction | 2 | 2.00 |
| 1.1 | &nbsp;&nbsp;Positioning among naive units | 2 | 1.01 |
| 1.2 | &nbsp;&nbsp;The three properties | 3 | 0.28 |
| 1.3 | &nbsp;&nbsp;Contributions | 3 | 0.67 |
| 2 | Methods | 4 | 1.83 |
| 3 | Experiments | 5 | 2.66 |
| 3.1 | &nbsp;&nbsp;The Shapes3D value hold-out | 5 | 1.13 |
| 3.2 | &nbsp;&nbsp;Baselines and the matching rule | 7 | 0.36 |
| 3.3 | &nbsp;&nbsp;The $n$-way ladder protocol | 7 | 0.21 |
| 3.4 | &nbsp;&nbsp;The deletion protocol | 7 | 0.26 |
| 3.5 | &nbsp;&nbsp;The CMAZE-T transition task | 7 | 0.66 |
| 4 | Results | 8 | 8.65 |
| 4.1 | &nbsp;&nbsp;Composition on Shapes3D | 8 | 0.91 |
| 4.2 | &nbsp;&nbsp;The $n$-way ladder | 9 | 2.41 |
| 4.3 | &nbsp;&nbsp;Deletion on Shapes3D | 11 | 1.73 |
| 4.4 | &nbsp;&nbsp;Deletion against retrained baselines | 13 | 0.33 |
| 4.5 | &nbsp;&nbsp;Refusal through the hole | 13 | 1.17 |
| 4.6 | &nbsp;&nbsp;Two ablations | 15 | 0.46 |
| 4.7 | &nbsp;&nbsp;Encoder adaptation and generative baselines | 15 | 0.47 |
| 4.8 | &nbsp;&nbsp;CMAZE-T conditions and relational coding | 16 | 1.15 |
| 5 | Discussion | 17 | 2.06 |
| 5.1 | &nbsp;&nbsp;Limitations | 17 | 0.75 |
| 5.2 | &nbsp;&nbsp;A planned CMAZE suite | 17 | 0.13 |
| 5.3 | &nbsp;&nbsp;The zero-dynamics clue | 18 | 0.30 |
| 5.4 | &nbsp;&nbsp;The Latent Dynamics Model class | 18 | 0.43 |
| 5.5 | &nbsp;&nbsp;Summary | 18 | 0.20 |
| 5.6 | &nbsp;&nbsp;Physics follow-up | 19 | 0.21 |
|  | &nbsp;&nbsp;AI use statement | 19 | 0.12 |
|  | &nbsp;&nbsp;Ethics statement | 19 | 0.15 |
|  | &nbsp;&nbsp;Reproducibility statement | 19 | 0.19 |
|  | References | 19 | 5.33 |
|  | Appendices | 25 | 0.12 |
| A | Reporting conventions | 25 | 0.41 |
| B | Tab.~\ref{tab:t7}: what it does and does not claim | 25 | 1.13 |
| C | The $K=5$ endpoint of the sweep | 26 | 0.58 |
| D | The $K$-curve: composition and the memory footprint against $K$ | 27 | 1.59 |
| E | Deletion on Shapes3D: two implementations, and the failure they agree on | 28 | 0.96 |
| F | The earlier composition cell: byte and parameter match, and the ceiling ladder | 29 | 0.95 |
| G | Deletion comparators in full (fit pool, 0 test runs) | 30 | 0.66 |
| H | Light encoder adaptation in full | 31 | 0.44 |
| I | The encoder loop in full (shortened-decoder instrument, and the longer leg) | 31 | 1.16 |
| J | The zero-dynamics identity: conditions and provenance | 33 | 0.73 |
| K | The retention law on the field-mode n-CLU in full | 33 | 0.85 |
| L | Prior derivations | 34 | 1.31 |
| M | What an evaluation must contain: the ten requirements | 35 | 0.48 |
| N | Maze world-model rows | 36 | 1.40 |
| O | The encoder: adaptation and the unfreezing decomposition | 37 | 2.11 |
| P | The two laws on this study's n-CLU, stated in full | 39 | 1.86 |
| Q | Further limitations | 41 | 0.82 |
| R | Placeholder index | 42 | 0.29 |
| S | Declared not-runs | 42 | 0.91 |

## DRAFT-v2-style, before: 44 pages; main text (Introduction -> References) 17.71 pages (Introduction on p. 2, References on p. 19)

| # | heading | page | len (pages) |
|---|---|---|---|
| 1 | Introduction: a naive unit, and why its three properties matter | 2 | 2.87 |
| 1.1 | &nbsp;&nbsp;Positioning: a naive unit, compared with naive units | 2 | 1.01 |
| 1.2 | &nbsp;&nbsp;Why the three properties matter: world models need all three at once | 3 | 0.38 |
| 1.3 | &nbsp;&nbsp;What we show, and at what thickness | 3 | 1.44 |
| 2 | The object, the judge and the memory footprint | 4 | 1.96 |
| 3 | The n-CLU on Shapes3D: the single design at $\Kstar$ | 6 | 8.74 |
| 3.1 | &nbsp;&nbsp;Composition at $\Kstar$: rows 3 and 4, three scored seeds | 7 | 1.06 |
| 3.2 | &nbsp;&nbsp;The $n$-way ladder at $\Kstar$, with its inference note | 8 | 3.21 |
| 3.3 | &nbsp;&nbsp;Deletion on the single design: value, conjunction, item | 11 | 1.92 |
| 3.4 | &nbsp;&nbsp;What the baselines can do instead: retrain without, or approximate | 13 | 0.55 |
| 3.5 | &nbsp;&nbsp;Refusal after deletion: the hole, a separate mechanism | 14 | 0.42 |
| 3.6 | &nbsp;&nbsp;Two ablations, each keeping one property | 14 | 0.43 |
| 3.7 | &nbsp;&nbsp;The encoder: frozen by design, lightly adapted; the encoder loop licenses no n-CLU claim | 15 | 0.49 |
| 4 | The same properties on a maze | 15 | 1.67 |
| 5 | Limitations of the naive unit, and a leaving note | 17 | 1.76 |
| 5.1 | &nbsp;&nbsp;What the naive unit does not do (measured, not hedged) | 17 | 0.76 |
| 5.2 | &nbsp;&nbsp;The clue: the n-CLU is the zero-dynamics limit of a settling unit | 18 | 0.30 |
| 5.3 | &nbsp;&nbsp;The theory we carry: the Latent Dynamics Model class | 18 | 0.42 |
| 5.4 | &nbsp;&nbsp;Leaving note | 18 | 0.20 |
| 6 | Follow-up: a full evaluation of the physics CLU | 19 | 0.23 |
|  | &nbsp;&nbsp;AI use statement | 19 | 0.12 |
|  | &nbsp;&nbsp;Ethics statement | 19 | 0.15 |
|  | &nbsp;&nbsp;Reproducibility statement | 19 | 0.21 |
|  | References | 19 | 5.29 |
|  | Appendices | 25 | 0.12 |
| A | Reporting conventions | 25 | 0.41 |
| B | Tab.~\ref{tab:t7}: what it does and does not claim | 25 | 1.13 |
| C | The $K=5$ endpoint of the sweep | 26 | 0.58 |
| D | The $K$-curve: composition and the memory footprint against $K$ | 27 | 1.60 |
| E | Deletion on Shapes3D: two implementations, and the failure they agree on | 28 | 0.97 |
| F | The earlier composition cell: byte and parameter match, and the ceiling ladder | 29 | 1.00 |
| G | Deletion comparators in full (fit pool, 0 test runs) | 30 | 0.64 |
| H | Light encoder adaptation in full | 31 | 0.20 |
| I | The encoder loop in full (shortened-decoder instrument, and the longer leg) | 31 | 0.26 |
| J | The zero-dynamics identity: conditions and provenance | 31 | 1.92 |
| K | The retention law on the field-mode n-CLU in full | 33 | 0.89 |
| L | Prior derivations | 34 | 1.50 |
| M | What an evaluation must contain: the ten requirements | 36 | 0.43 |
| N | Maze world-model rows | 36 | 2.14 |
| O | The encoder: adaptation and the unfreezing decomposition | 38 | 1.96 |
| P | The two laws on this study's n-CLU, stated in full | 40 | 1.87 |
| Q | Further limitations | 42 | 0.24 |
| R | Placeholder index | 42 | 0.78 |
| S | Declared not-runs | 43 | 0.41 |

## DRAFT-v2-style, after: 44 pages; main text (Introduction -> References) 17.86 pages (Introduction on p. 2, References on p. 19)

| # | heading | page | len (pages) |
|---|---|---|---|
| 1 | Introduction | 2 | 2.13 |
| 1.1 | &nbsp;&nbsp;Positioning among naive units | 2 | 1.01 |
| 1.2 | &nbsp;&nbsp;The three properties | 3 | 0.38 |
| 1.3 | &nbsp;&nbsp;Contributions | 3 | 0.69 |
| 2 | Methods | 4 | 1.11 |
| 3 | Experiments | 5 | 3.07 |
| 3.1 | &nbsp;&nbsp;The Shapes3D value hold-out | 5 | 0.31 |
| 3.2 | &nbsp;&nbsp;Baselines and the matching rule | 5 | 0.36 |
| 3.3 | &nbsp;&nbsp;The $n$-way ladder protocol | 5 | 0.95 |
| 3.4 | &nbsp;&nbsp;The deletion protocol | 6 | 1.12 |
| 3.5 | &nbsp;&nbsp;The CMAZE-T transition task | 8 | 0.31 |
| 4 | Results | 8 | 8.98 |
| 4.1 | &nbsp;&nbsp;Composition on Shapes3D | 8 | 2.54 |
| 4.2 | &nbsp;&nbsp;The $n$-way ladder | 10 | 1.74 |
| 4.3 | &nbsp;&nbsp;Deletion on Shapes3D | 12 | 1.14 |
| 4.4 | &nbsp;&nbsp;Deletion against retrained baselines | 13 | 0.85 |
| 4.5 | &nbsp;&nbsp;Refusal through the hole | 14 | 0.60 |
| 4.6 | &nbsp;&nbsp;Two ablations | 15 | 0.42 |
| 4.7 | &nbsp;&nbsp;Encoder adaptation and generative baselines | 15 | 0.48 |
| 4.8 | &nbsp;&nbsp;CMAZE-T conditions and relational coding | 16 | 1.16 |
| 5 | Discussion | 17 | 2.10 |
| 5.1 | &nbsp;&nbsp;Limitations | 17 | 0.75 |
| 5.2 | &nbsp;&nbsp;A planned CMAZE suite | 18 | 0.12 |
| 5.3 | &nbsp;&nbsp;The zero-dynamics clue | 18 | 0.29 |
| 5.4 | &nbsp;&nbsp;The Latent Dynamics Model class | 18 | 0.41 |
| 5.5 | &nbsp;&nbsp;Summary | 18 | 0.22 |
| 5.6 | &nbsp;&nbsp;Physics follow-up | 19 | 0.22 |
|  | &nbsp;&nbsp;AI use statement | 19 | 0.12 |
|  | &nbsp;&nbsp;Ethics statement | 19 | 0.15 |
|  | &nbsp;&nbsp;Reproducibility statement | 19 | 0.21 |
|  | References | 19 | 5.14 |
|  | Appendices | 25 | 0.12 |
| A | Reporting conventions | 25 | 0.41 |
| B | Tab.~\ref{tab:t7}: what it does and does not claim | 25 | 1.13 |
| C | The $K=5$ endpoint of the sweep | 26 | 0.58 |
| D | The $K$-curve: composition and the memory footprint against $K$ | 27 | 1.60 |
| E | Deletion on Shapes3D: two implementations, and the failure they agree on | 28 | 0.97 |
| F | The earlier composition cell: byte and parameter match, and the ceiling ladder | 29 | 1.00 |
| G | Deletion comparators in full (fit pool, 0 test runs) | 30 | 0.64 |
| H | Light encoder adaptation in full | 31 | 0.20 |
| I | The encoder loop in full (shortened-decoder instrument, and the longer leg) | 31 | 0.26 |
| J | The zero-dynamics identity: conditions and provenance | 31 | 1.92 |
| K | The retention law on the field-mode n-CLU in full | 33 | 0.89 |
| L | Prior derivations | 34 | 1.50 |
| M | What an evaluation must contain: the ten requirements | 36 | 0.43 |
| N | Maze world-model rows | 36 | 2.14 |
| O | The encoder: adaptation and the unfreezing decomposition | 38 | 1.96 |
| P | The two laws on this study's n-CLU, stated in full | 40 | 1.87 |
| Q | Further limitations | 42 | 0.24 |
| R | Placeholder index | 42 | 0.78 |
| S | Declared not-runs | 43 | 0.41 |

