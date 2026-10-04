# RESIDUALS — rendered-source token scan, before and after

Command (per draft): `python3 pw1_draft_sweeps/scan.py <DRAFT>.tex --lines` (word-boundary regexes; LaTeX comments and `\label`/`\ref`/`\cite` keys excluded; case-insensitive except A1, A10, bare CLU, `\S`).

| token | v2 before | v2 after | style before | style after |
|---|---|---|---|---|
| rival | 194 | 0 | 194 | 0 |
| store | 387 | 0 | 388 | 0 |
| gate | 15 | 0 | 15 | 0 |
| artifact/artefact | 16 | 0 | 17 | 0 |
| pre-register | 38 | 0 | 38 | 0 |
| A1 | 14 | 0 | 14 | 0 |
| A10 | 12 | 0 | 12 | 0 |
| C-512 | 2 | 0 | 2 | 0 |
| improved | 2 | 0 | 2 | 0 |
| config | 52 | 26 | 52 | 26 |
| spoke | 0 | 0 | 0 | 0 |
| hub | 0 | 0 | 0 | 0 |
| wave | 0 | 0 | 0 | 0 |
| charter | 3 | 0 | 3 | 0 |
| ledger | 25 | 0 | 26 | 0 |
| bar k | 0 | 0 | 0 | 0 |
| scored run | 24 | 0 | 24 | 0 |
| budget | 58 | 0 | 58 | 0 |
| spend/spent | 4 | 0 | 4 | 0 |
| price/priced/pricing | 8 | 0 | 8 | 0 |
| buy | 7 | 0 | 8 | 0 |
| pay | 7 | 0 | 7 | 0 |
| cost | 20 | 8 | 20 | 8 |
| cheap | 3 | 1 | 3 | 1 |
| expensive | 1 | 0 | 1 | 0 |
| t_pay | 0 | 0 | 0 | 0 |
| corner | 106 | 0 | 106 | 0 |
| held-out composition | 0 | 0 | 0 | 0 |
| oblong | 0 | 0 | 0 | 0 |
| exact judge | 14 | 0 | 14 | 0 |
| exact | 149 | 114 | 149 | 114 |
| 10x | 0 | 0 | 0 | 0 |
| landscape_removed | 4 | 0 | 4 | 0 |
| landscape_shuffled | 12 | 0 | 12 | 0 |
| random_atoms | 9 | 0 | 9 | 0 |
| Naive CLU | 4 | 0 | 4 | 0 |
| CLU (bare) | 9 | 3 | 9 | 3 |
| \S | 43 | 1 | 44 | 1 |
| Section~ | 0 | 0 | 0 | 0 |
| Figure~ | 10 | 0 | 10 | 0 |
| Table~ | 87 | 1 | 87 | 1 |
| Appendix~ | 31 | 0 | 31 | 0 |
| Sec.~ | 0 | 41 | 0 | 42 |
| Fig.~ | 0 | 10 | 0 | 10 |
| Tab.~ | 0 | 86 | 0 | 86 |
| App.~ | 0 | 31 | 0 | 31 |
| Eq.~ | 0 | 0 | 0 | 0 |
| Alg.~ | 0 | 0 | 0 | 0 |
| bare \ref | 48 | 7 | 49 | 7 |

## Every residual line after the sweep

### DRAFT-v2

```
rival                     0
store                     0
gate                      0
artifact/artefact         0
pre-register              0
A1                        0
A10                       0
C-512                     0
improved                  0
config                   26
      L649 [configuration] measured} (a new TEST configuration would be a new test run, and the \texttt{composition} target is
      L1084 [configuration] compressed baseline: at this configuration the n-CLU's read size holds more latents than the split
      L1164 [configuration] relational baselines is sizeable only on seed 2 (+0.165). We report the configuration as evidence about
      L1188 [configuration] validation $n$-way against the n-CLU's 0.640 / 0.730 / 0.730, so no CMAZE-O configuration can pass
      L1194 [configurations] other two CMAZE-O configurations fail validation with their own links (the autoencoder latent at
      L1203 [configuration] n-CLU is ahead on three of six seeds and its relational configuration is a request-coding result, and
      L1204 [configuration] the CMAZE-O control check with its withdrawn margin. The run count, the configuration ladder, the fallback
      L1236 [configuration] $\Kstar$ is not the point of smallest footprint; and on the one configuration at $K\ge36$ whose composition
      L1324 [configuration] baselines is small and seed-dependent, and its best configuration is a request coding any learner can be
      L1399 [configurations] \paragraph{Seeds and configurations.} Seed count and configuration count are separate fields. The
      L1399 [configuration] \paragraph{Seeds and configurations.} Seed count and configuration count are separate fields. The
      L1400 [configurations] phrase ``nine seeds'' never applies: where nine legs appear they are three configurations times
      L1677 [configurations] implementation & seeds & configurations & test runs & band \\
      L1689 [configurations] shared.} Seeds and configurations are separate fields; the nine legs of row 3 are 3 configurations
      L1689 [configurations] shared.} Seeds and configurations are separate fields; the nine legs of row 3 are 3 configurations
      L1696 [configurations] implementation & seeds & configurations & test runs & band \\
      L1886 [configurations] \caption{\textbf{Encoder adaptation at $\Kstar$, TEST, 3 seeds $\times$ 3 configurations (nine
      L2099 [configurations] per row, (b), (c)~six seeds and two configurations.} (a)~Validity under the engine's judge at each seed for the n-CLU (circles) and for the
      L2586 [configurations] unseen $n$-way & 3 & \textbf{n-CLU row under asymmetric inputs only} (the margin does not survive symmetric inputs, Tab.~\ref{}): n-CLU 0.7246 (0.800 / 0.566 / 0.808) against the best matched baseline
      L2638 [configuration] steps), a diagnostic and not a matched baseline. Every panel is re-read from this configuration's
      L2742 [Configuration] \paragraph{Configuration ladder and run count.} Every configuration was measured on validation first,
      L2742 [configuration] \paragraph{Configuration ladder and run count.} Every configuration was measured on validation first,
      L2744 [configurations] 53 configurations on three seeds left 25/53 eligible under the control check, and the crossover rule picked
      L2758 [configuration] configuration passes validation against converged baselines; a field configuration would need $n$-way
      L2758 [configuration] configuration passes validation against converged baselines; a field configuration would need $n$-way
      L2759 [configuration] of at least 1.02 to clear the 0.99 MLP, and seeding a configuration that fails validation is
spoke                     0
hub                       0
wave                      0
charter                   0
ledger                    0
bar k                     0
scored run                0
budget                    0
spend/spent               0
price/priced/pricing      0
buy                       0
pay                       0
cost                      8
      L244 [cost] \item \textbf{Exact deletion on the n-CLU and on its address book, with its cost and its limitation
      L735 [cost] \multicolumn{8}{l}{\textbf{cost}, seconds per seed (0 / 1 / 2)} \\
      L812 [cost] $\Kstar$ object two holes cost 2 bytes, read 0 data rows, take 0.0000 s and refit nothing, in
      L825 [cost] \caption{\textbf{Refusal DISPLAY rows. A display is never a result.} Our side is a cost line
      L854 [costs] 4,096 data rows read, 521,916 bytes rewritten and 10.48 s, where the coupling hole costs one
      L891 [cost] the memory cost printed in Tab.~\ref{}.
      L1326 [cost] memory and time cost of exactness, in bytes and in the book's refit. Switched on, the dynamics lower composition at
      L1823 [cost] \caption{\textbf{Deletion cost for the data-matched baselines: retrain without the rows, on the n-CLU's
cheap                     1
      L1230 [cheap] \item \textbf{Exact deletion is not cheap in memory or time.} The book's exact statistic is 0.97 GB,
expensive                 0
t_pay                     0
corner                    0
held-out composition      0
oblong                    0
exact judge               0
exact                   114
      L75 [Exactly] \title{naive-CLU: Exactly Deletable Compositional Generation\\ on a Frozen Encoder}
      L91 [exactly] \emph{forget} specific training rows exactly. Compositional generation is a crowded area
      L92 [exact] and exact unlearning is an established one. We work at the level of a single \emph{naive unit} and
      L105 [exactly] controls collapsing to exactly 0.0000.
      L117 [exact] Deletion is exact on the n-CLU and on its address book: after a value, a conjunction or a random
      L119 [exact] most \sci{5.38}{-07}, book statistic deviation 0). It is not free: the book's exact statistic is
      L137 [exactly] dynamical successor: the n-CLU is exactly the zero-dynamics limit of a settling unit, with
      L167 [Exact] Exact unlearning is also established, for $k$-means (Ginart et al., 2019\nocite{}), for bilinear
      L181 [exact] false positives and none exact; no work among the 3,038 combines frozen-decoder,
      L186 [exact] \emph{generative} $\wedge$ \emph{exact deletion}, returns zero works on the screen.
      L193 [exactly] exactly unlearnable by splitting its data into shards and retraining the unit on the shard that
      L195 [exact] whose only exact route is the retrain of Sec.~\ref{}.
      L210 [exact] instability, and give no rigorous error analysis. Our route is exact for the opposite reason:
      L223 [Exact] values that it has seen only separately~\citep{}. \textbf{Exact deletion}: after a set of training rows
      L225 [exactly] those rows. The three pull against each other: a lookup table deletes exactly but composes
      L226 [exactly] nothing it has not written; a learned generator composes but can delete exactly only by retraining
      L228 [exact] approximate unlearning~\citep{} is approximate by construction, not exact (Sec.~\ref{}).
      L244 [Exact] \item \textbf{Exact deletion on the n-CLU and on its address book, with its cost and its limitation
      L249 [exact] training row, but the book's exact statistic is 0.97 GB, the size of its fit data, and its refit
      L252 [exactly] \item \textbf{The same properties on a maze.} Deletion transfers exactly; composition on routes is
      L332 [exact] \paragraph{The retained statistic and exact deletion.} Beside the content the n-CLU keeps, per
      L335 [exact] statistic. The address book keeps its own exact statistic, per request tuple the count and first
      L338 [exactness] (Sec.~\ref{}). We use one definition of exactness throughout, and state its scope
      L342 [Exact] \textsc{Exact is scoped to the n-CLU and its address book.} After deleting a set of rows, the n-CLU's
      L419 [exactly] all four landscape controls read exactly 0.000 on both rows on all three seeds, with the
      L422 [exact] Every baseline is read-path matched (130,433 to 131,891 parameters) and fitted on the n-CLU's exact
      L530 [exactly] rungs $n\ge3$ are exactly constant by construction.
      L532 [exactly] 13 of 15 read exactly 0.0000 and two ($n=3$, seeds 0 and 1) read 0.0083 on
      L546 [exactly] families fitted on the rung's first 32,768 rows, exactly the n-CLU's content rows, at the selected
      L587 [exactly] \multicolumn{7}{l}{\emph{n-CLU-content set: the rung's first 32,768 rows, exactly the n-CLU's content rows; 5,120 steps, ten times 51,200}} \\
      L612 [exactly] $r=-0.96$ to $-0.98$, and every arm not fitted on it is exactly constant.
      L627 [exactly] 32,768 content rows, favour the baselines. They do not: fitted on exactly the n-CLU's content rows,
      L669 [exact] rung's own 131,072-row pool, the n-CLU's exact rows, at 20,480 steps, or at 204,800 where the
      L680 [Exact] \subsection{Exact deletion on the single design: value, conjunction, item}
      L745 [exact] \noindent Measurements (a) and (b) are exact on every scope and seed, and (d)'s check is exact too:
      L745 [exact] \noindent Measurements (a) and (b) are exact on every scope and seed, and (d)'s check is exact too:
      L748 [exact] book's refit reads no pool row but reads its exact statistic, 967,304,800 B (0.97 GB). The book sees
      L767 [Exactness] \paragraph{The probe that bounds the claim.} Exactness is n-CLU-scoped in a way that is
      L786 [exact] A learned baseline's only exact deletion route is to retrain from scratch without the rows. We ran
      L796 [exact] collateral damage the baselines' crude retrain is cleaner than the n-CLU's exact deletion. Nor is the
      L799 [exactness] is exactness by construction and no pool row read. Two caveats travel with these rows: the baselines
      L810 [exact] Refusal is a second exact mechanism on the same object: a coupling table with \emph{holes}.
      L856 [exact] \noindent The operation that is both exact on bytes and free on behaviour is the hole, not
      L859 [exactly] while punching a hole refuses at 1.0000 and changes it by exactly $+0.0000$.
      L860 [exactness] The claim is exactness, never erasure without loss.
      L867 [exact] statistic, which is the requirement it does not meet and the reason it has no exact deletion. At the
      L906 [exact] on rows 3--4, render no exact-match request on the generation table on any seed, and there score at
      L946 [exact] The n-CLU's exact deletion was run inside the three declared test runs on CMAZE-2:
      L956 [Exact] \caption{\textbf{Exact deletion on the maze, CMAZE-2, TEST, 3 of 3 scored seeds.} Delete the
      L977 [Exact] \noindent Exact is not free here either. As declared in advance, composition on the surviving
      L991 [exactness] three seeds, so the claim is n-CLU-scoped exactness and nothing more.
      L999 [exactly] of 1,449 optimal routes picked up exactly one key, so there was no key order for the second rule
      L1088 [exactly] other mazes' routes, the published tripwire, reads exactly 0.0000 at $r=1$ and $r=2$ on every
      L1133 [exact] control stays below the n-CLU on every seed (highest 0.495, seed 5), and deletion stays exact on all
      L1175 [Exact] printed in Tab.~\ref{} and is not a comparison of like with like. Exact deletion holds on
      L1178 [exactly] n-CLU reads exactly as the n-CLU fitted without the rows, in 0.09 to 0.88 s with no gradient step,
      L1196 [Exact] no single failing link. Exact deletion also runs on the field design (relative change of the statistic 0.0 on
      L1225 [exact] is exact on the n-CLU and on the address book, but deleting the red sphere lowers composition on
      L1230 [Exact] \item \textbf{Exact deletion is not cheap in memory or time.} The book's exact statistic is 0.97 GB,
      L1230 [exact] \item \textbf{Exact deletion is not cheap in memory or time.} The book's exact statistic is 0.97 GB,
      L1233 [Exact] \item \textbf{Exact requires a large number of bytes.} The deletable object at $\Kstar$ is 4.0768 to 4.0897
      L1248 [exact] in this feature space, render no exact-match request on the generation table and score at the level
      L1284 [exactly] $z_{\text{req}}$ exactly at $\lambda=0$. Switching the dynamics on lowers composition at every
      L1320 [exact] The naive unit provides exact deletion by subtraction and, at a stated partition, composition on
      L1326 [exactness] memory and time cost of exactness, in bytes and in the book's refit. Switched on, the dynamics lower composition at
      L1350 [Exact] involves no human subjects and no personal data. Exact deletion is relevant to data-protection
      L1351 [exactness] requests; we claim exactness on the n-CLU and its address book only and no certified unlearning, and
      L1352 [exact] we state in Sec.~\ref{} a case in which exact deletion of a conjunction changes what the
      L1457 [exact] (0.1204), and neither renders an exact-match request on any seed.}
      L1501 [exact] exact-match requests on the generation table as well; in the words of that experiment's own record, a
      L1666 [Exact] \section{Exact deletion: three implementations, and the failure they agree on}
      L1671 [Exact] \caption{\textbf{Exact (n-CLU-scoped) deletion, implementation 1: the maze n-CLU (CMAZE-2).} The
      L1680 [exactly] 1. maze n-CLU (CMAZE-2, TEST) & 3 & 1 & 3 & \sci{7.522}{-16} .. \sci{8.826}{-16} on the statistic; projector exactly 0.000 \\
      L1688 [Exact] \caption{\textbf{Exact (n-CLU-scoped) deletion on Shapes3D, implementations 2 and 3, bands not
      L1710 [exactness] its own floor and is not resolvable at all, the best available outcome for an exactness claim. The
      L1724 [exactly] exactly 16,384 B between the serialised object and the honest total is declared, not netted.
      L1727 [exactness] \paragraph{The failure agreement.} The exactness assertion failed first, on TEST seed 2, on the
      L1740 [exactness] (rank over width 0.31 to 0.21); it is the only place the exactness claim removes content.
      L1745 [exactness] reading 120,048 rows, exactness \sci{1.455}{-11}, survivor $\Delta$ $+0.1250$, $\cos_{\min}$
      L1746 [exactness] 0.183704. Red sphere: 5,507 rows, 0.6205 s, reference 10.843 s / 125,565 rows, exactness
      L1748 [exactness] 0.2554 s, reference 11.446 s / 129,761 rows, exactness \sci{1.455}{-11}, survivor $\Delta$
      L1801 [exactly] of the rendered features; the book's count and first-moment statistic deviate by exactly 0), per
      L1910 [exact] \noindent Factor-probe margins over random initialisation, aggregate / exact-match, per seed, never
      L2119 [exactly] coding, not from the n-CLU. On seed 4 the random-atoms control also equals the n-CLU exactly
      L2216 [Exact] \paragraph{Exact deletion on CMAZE-2b in full.} Tab.~\ref{} is the CMAZE-2b
      L2223 [Exact] checkout reproduces both. Exact is not light in extent: the exact-head downdate changes every atom
      L2223 [exact] checkout reproduces both. Exact is not light in extent: the exact-head downdate changes every atom
      L2228 [exactness] 0.4404, 0.5638 $\to$ 0.5319), so the claim is n-CLU-scoped exactness and nothing more. Speed is not
      L2230 [exact] content and head from the survivors. The difference is that it is exact without keeping the items,
      L2238 [Exact] \caption{\textbf{Exact deletion on the maze, CMAZE-2b, 3 seeds, 0 test runs.} This table
      L2254 [exact] exact-head downdate: seconds & 3 & 2.042 / 2.328 / 2.181 \\
      L2303 [exactly] pushed off the span by $\delta=0.10$ the settle moves by exactly $\delta/(1+\beta)$, measured
      L2307 [exactly] cross-well term: at $\lambda=0$ the displacement is exactly 0.0; at
      L2328 [exact] is verified over the whole band, and $1+\gamma/(2-\gamma)$ is the exact log-derivative (numeric
      L2331 [exactly] margin 0.0049 / 0.0047 / 0.0043 over the naive $-1$); in that branch the slow multiplier is exactly
      L2334 [exactly] (\texttt{d14\_confinement}, exactly 0.0 on 60 cells by construction) is never quoted; the
      L2359 [exact] B4 & 3 & \ldots against the exact inversion & 0.3782 & 0.3784 & 0.3834 \\
      L2381 [exact] half-life form is 2.4\% off already at $h=0.30h^\star$ and is never quoted, only the exact root.
      L2383 [exactly] The seed-0 leg replicates bit-exactly from a clean checkout; two of the three declared runs
      L2405 [exactly] against composition. An 8-byte rule ties the field-mode n-CLU exactly on the two symbolic refusal rows,
      L2420 [exact] \paragraph{Memory-level exact deletion in a designed memory (\textsc{confirmed [prior
      L2421 [exact] derivation]}).} With an order-independent waitlist, memory-level deletion is exact at every load
      L2433 [exact] rollout prediction on this n-CLU).} On the exact one-step Jacobian the emergent retention
      L2444 [exactly] basin displacement is exactly zero off the coupling graph (0 of 4656 off-graph entries nonzero
      L2487 [exact] memory 6.3 KB against 6.00 MB at $T=1024$; exact only at zero damping.
      L2492 [exact] read size, a memory cannot have all three of exact value fidelity, amplitude-independent address
      L2509 [exactly] 39\% exactly optimal).
      L2512 [exact] \item Run a matched-bytes stitching tripwire with a fragment-length sweep before adoption: exact-key
      L2592 [exactness] deletion, value & 3 & colour 1; n-CLU-scoped exactness $S_k$ relative change \sci{7.80}{-16} / \sci{1.22}{-15} / \sci{1.09}{-15}, atoms change 0; downdated read equal to fitted-without: 0.4369 / 0.425
      L2639 [exactly] persisted objects; the re-read reproduces all 132 scored TEST arm scores exactly, over seeds 0--5.
      L2702 [exactness] deletion & 6 & base n-CLU, words + observation: n-CLU-scoped exactness on every seed (relative statistic change at most \sci{1.94}{-15}, atoms unchanged, reads after the downdate identical to the n-CL
      L2734 [exactness] deletion, conj. & 3 & 2240 / 1435 / 2168 rows deleted; relative change 0.0; deleted side 0.5800 / 0.8000 / 0.7067 $\to$ 0.0000; other colours 0.6457 $\to$ 0.6343, 0.6880 $\to$ 0.6800, 0.7160 $\to$ 0.7
      L2833 [exact] On an exact quadratic, where the law holds by construction, the same measurement chain recovers it
      L2859 [exact] law. Two consequences derived before measurement were then exact: the refusal horizon is $19.9316$
10x                       0
landscape_removed         0
landscape_shuffled        0
random_atoms              0
Naive CLU                 0
CLU (bare)                3
      L1274 [CLU] The n-CLU we evaluate is the zero-dynamics limit of the CLU: the same content, the same wells,
      L1333 [CLU] \section{Follow-up: a full evaluation of the physics CLU}
      L1337 [CLU] The follow-up is a full evaluation of the settling (physics) CLU on the three properties of
\S                        1
      L134 [\S] (\S\ref{}, Table~\ref{}) replaces the earlier asymmetric-input sentence; no
Section~                  0
Figure~                   0
Table~                    1
      L134 [Table~\ref] (\S\ref{}, Table~\ref{}) replaces the earlier asymmetric-input sentence; no
Appendix~                 0
Sec.~                    41
      L152 [Sec.~\ref] (Sec.~\ref{}). We do not claim the best compositional-generation system or the best
      L190 [Sec.~\ref] three months is incomplete (Sec.~\ref{}).
      L195 [Sec.~\ref] whose only exact route is the retrain of Sec.~\ref{}.
      L228 [Sec.~\ref] approximate unlearning~\citep{} is approximate by construction, not exact (Sec.~\ref{}).
      L241 [Sec.~\ref] to +21.936 over the transformer (Sec.~\ref{}). On every rung of an $n$-way ladder it
      L243 [Sec.~\ref] mean (Sec.~\ref{}); for $n\ge3$ the rungs are a training-data axis and not a depth axis.
      L248 [Sec.~\ref] about the value it shares with the survivors (Sec.~\ref{}). The deletion reads no
      L251 [Sec.~\ref] separate, structural mechanism, a hole in a coupling table (Sec.~\ref{}).
      L254 [Sec.~\ref] matched-bytes lookup floor more than it lifts the n-CLU (Sec.~\ref{}). On a transition
      L259 [Sec.~\ref] (Sec.~\ref{}). We offer the transition instrument itself as a minor contribution.
      L261 [Sec.~\ref] forget; an item memory forgets and cannot compose (Sec.~\ref{}). The naive unit's
      L263 [Secs.~\ref] (Secs.~\ref{}--\ref{}).
      L273 [Sec.~\ref] (Sec.~\ref{}). The rows still pending a run are marked placeholders whose text names the row.
      L338 [Sec.~\ref] (Sec.~\ref{}). We use one definition of exactness throughout, and state its scope
      L396 [Sec.~\ref] Sec.~\ref{} on the Shapes3D value hold-out~\citep{}, with the partition fixed on validation
      L488 [Sec.~\ref] \paragraph{A second, earlier baseline set.} The ladder (Sec.~\ref{}) also prints an earlier set
      L544 [Sec.~\ref] combination-split pool that carries the held-out values (Sec.~\ref{}), at 5,040 optimiser
      L690 [Sec.~\ref] hole for $c$ (Sec.~\ref{}). Every deletion row therefore prints four measurements
      L781 [Sec.~\ref] is smallest. This is why refusal needs its own mechanism (Sec.~\ref{}).
      L910 [Sec.~\ref] the study has no functioning generative baseline (Sec.~\ref{}).
      L941 [Sec.~\ref] observation instruments whose rows are in Sec.~\ref{}.
      L950 [Sec.~\ref] world-model instruments of Sec.~\ref{} also do), and it replicates from a clean
      L984 [Sec.~\ref] The replication on CMAZE-2b, the environment of the verb result (Sec.~\ref{}), is its own
      L1101 [Sec.~\ref] The two instruments below are the maze counterpart of Sec.~\ref{}: they ask the n-CLU to
      L1220 [Sec.~\ref] CMAZE-2b the controls again fail to collapse (Sec.~\ref{}).
      L1223 [Sec.~\ref] (Sec.~\ref{}, Tab.~\ref{}).
      L1228 [Sec.~\ref] baseline without the same rows is cleaner on collateral damage (Sec.~\ref{},
      L1229 [Sec.~\ref] Sec.~\ref{}).
      L1243 [Sec.~\ref] per held-out combination from ten to thirty times its steps (Sec.~\ref{}).
      L1338 [Sec.~\ref] Sec.~\ref{}, same judges, memory footprint and matching, with this paper's n-CLU as its zero-dynamics
      L1352 [Sec.~\ref] we state in Sec.~\ref{} a case in which exact deletion of a conjunction changes what the
      L1358 [Sec.~\ref] defined in Sec.~\ref{}; every table states its seeds, condition and pins; declared
      L1469 [Sec.~\ref] \multicolumn{5}{p{15cm}}{\emph{LACE and PoE are added beside the baselines above, which are not replaced; each runs at its own declared settings (LACE 31.08$\times$, PoE 27.135$\times$ the 524,288 B r
      L1662 [Sec.~\ref] resolve the validation plateau (Sec.~\ref{}); the values are those of
      L2141 [Sec.~\ref] (Tab.~\ref{}); the before/after contrast is stated in prose (Sec.~\ref{}).}
      L2217 [Sec.~\ref] replication of Sec.~\ref{}: its own table, never merged with Tab.~\ref{}, and
      L2283 [Sec.~\ref] refusal score on a route request; the Shapes3D coupling hole (Sec.~\ref{}) is a structural
      L2428 [Sec.~\ref] paper adopts in Sec.~\ref{}: an n-CLU-level guarantee only, the frozen encoder and any
      L2567 [Sec.~\ref] (Sec.~\ref{}); the symmetric-input rows are Tab.~\ref{}, where a matched
      L2895 [Sec.~\ref] (Tab.~\ref{}); LACE and PoE with the generation-table run (Sec.~\ref{},
      L2898 [Sec.~\ref] (Sec.~\ref{}). No figure is a placeholder figure and none carries an in-image banner.
Fig.~                    10
      L301 [Figs.~\ref] renderer of the result panels (Figs.~\ref{}--\ref{}), so
      L1113 [Fig.~\ref] (Fig.~\ref{}). Test mazes are generated at random, so a lookup of training mazes
      L2120 [Fig.~\ref] (0.9004, on every held-out tuple, Fig.~\ref{}); on seed 5 the best matched
      L2632 [Fig.~\ref] Fig.~\ref{}.}{fig:cmazet-per-arm}
      L2636 [Fig.~\ref] Fig.~\ref{}, for n-CLU-REL and its arms, and for the six baselines
      L2640 [Fig.~\ref] Under each $s_{t+1}$, its emitted facts, as in Fig.~\ref{}.}{fig:cmazet-rel-per-arm}
      L2644 [Fig.~\ref] Columns, and the emitted facts under each $s_{t+1}$, as in Fig.~\ref{}. For
      L2656 [Fig.~\ref] without; under each $s_{t+1}$, its emitted facts, as in Fig.~\ref{}. For the
      L2887 [Figs.~\ref] Figs.~\ref{}, \ref{} and~\ref{}. They are
      L2899 [Figs.~\ref] Figs.~\ref{}, \ref{}, \ref{}
Tab.~                    86
      L408 [Tab.~\ref] 12,203 B and 30,507 B; App.~\ref{}, Tab.~\ref{}).
      L415 [Tab.~\ref] Tab.~\ref{} is the scored result. Row 3 holds out 60 pair combinations and row 4 holds out
      L450 [Tab.~\ref] Tab.~\ref{}.}
      L484 [Tab.~\ref] itself is larger, and its memory footprint (Tab.~\ref{}) is printed, never
      L526 [Tab.~\ref] $n=6$ are the two ends. Tab.~\ref{} gives the n-CLU at $\Kstar$ on every rung and
      L551 [Tab.~\ref] and PoE are in Tab.~\ref{}. $n=2$: 60 combinations; $n\ge3$: the identical
      L616 [Tab.~\ref] On every rung and every seed the n-CLU's score in Tab.~\ref{} exceeds every baseline's
      L644 [Tab.~\ref] \emph{derived, not measured}: it is the $n=3$ n-CLU cell of Tab.~\ref{}, and no new
      L660 [Tab.~\ref] Tab.~\ref{}: \emph{``We beat the naive fair baselines at every $n$; the n-CLU's own 2-way
      L671 [Tab.~\ref] (Tab.~\ref{}). Open markers are the seeds; the line is their mean. The n-CLU's margin
      L691 [Tab.~\ref] (Tab.~\ref{}): (a) the n-CLU's content equals the content refit without the rows, a result; (b) the book
      L776 [Tab.~\ref] now also downdates the address book's statistics (measurement (b) of Tab.~\ref{}). The trade-off moved
      L788 [Tabs.~\ref] pool, at the baseline's selected schedule, three seeds (Tabs.~\ref{}
      L801 [Tab.~\ref] of Tab.~\ref{} travel with them; and approximate unlearning of the data-matched baselines was not
      L818 [Tab.~\ref] holed value is refused by construction, so refusal is measurement (c) of Tab.~\ref{}, a
      L874 [Tab.~\ref] the records give the two totals printed here and in Tab.~\ref{}, and they are
      L888 [Tab.~\ref] Tab.~\ref{} (App.~\ref{}) prints both ablations per seed. The pair shows
      L891 [Tab.~\ref] the memory cost printed in Tab.~\ref{}.
      L908 [Tab.~\ref] Tab.~\ref{} (App.~\ref{}). \emph{This is LACE and PoE as instantiated in
      L913 [Tab.~\ref] The generation table is a second Shapes3D instrument, never merged with Tab.~\ref{}: 72
      L916 [Tab.~\ref] nothing not-run; Tab.~\ref{},
      L948 [Tab.~\ref] the n-CLU refit without those rows. Tab.~\ref{} is the result. It was the
      L985 [Tab.~\ref] table, never merged with Tab.~\ref{}, and uses no test run (Tab.~\ref{},
      L985 [Tab.~\ref] table, never merged with Tab.~\ref{}, and uses no test run (Tab.~\ref{},
      L1008 [Tabs.~\ref] oracle stays at 1.0000 (Tabs.~\ref{} and~\ref{}, App.~\ref{}).
      L1096 [Tab.~\ref] (Tab.~\ref{}); the composition result on routes belongs to the verb.
      L1106 [Tabs.~\ref] instruments never share a table (Tabs.~\ref{}, \ref{}
      L1123 [Tab.~\ref] Tab.~\ref{} gives every input condition, and within a condition every arm receives the
      L1159 [Tab.~\ref] Tab.~\ref{}) reach 0.915 / 0.869 / 0.819 / 0.921 / 0.900 / 0.939, so the n-CLU's lead
      L1175 [Tab.~\ref] printed in Tab.~\ref{} and is not a comparison of like with like. Exact deletion holds on
      L1223 [Tab.~\ref] (Sec.~\ref{}, Tab.~\ref{}).
      L1234 [Tab.~\ref] times the reference (Tab.~\ref{}); across nine swept partitions the honest
      L1249 [Tab.~\ref] of our landscape controls (Tab.~\ref{}); the study compares naive units only and has no
      L1404 [Tab.~\ref] \section{Tab.~\ref{}: what it does and does not claim}
      L1406 [Tab.~\ref] The baselines of Tab.~\ref{} are matched on the read path at 130,514 (MLP), 130,721 (LSTM),
      L1412 [Tab.~\ref] in Tab.~\ref{} and is never matched. Baselines are matched on read-path
      L1419 [Tab.~\ref] generation-2 cache.} A separate instrument from Tab.~\ref{}, never merged with it:
      L1452 [Tab.~\ref] 2).} Distinct valid novel tuples per held-out combination, on the combinations of Tab.~\ref{} (upper block) and of
      L1453 [Tab.~\ref] every rung of Tab.~\ref{} (lower block). Neither is matched on the read path or on bytes;
      L1455 [Tab.~\ref] the generation table (Tab.~\ref{}) their aggregates, 0.3032 and 0.1968, sit at the
      L1489 [Tab.~\ref] learned baselines on rows 3--4 and on every ladder rung (Tab.~\ref{}), 3 seeds; no
      L1511 [Tab.~\ref] baseline by +3.780 / +6.872 / +3.697 (mean +4.783). On the ladder, Tab.~\ref{} prints the $K=5$
      L1521 [Tab.~\ref] Tab.~\ref{} (each rung's own 131,072 rows) and against the strongest baseline of the earlier
      L1523 [Tab.~\ref] Tab.~\ref{}:} for $n\ge3$ the rungs differ in fit-pool coverage, not in composition
      L1575 [Tab.~\ref] honest four-part total is U-shaped. The $n=1$ middle rows are superseded by Tab.~\ref{},
      L1605 [Tab.~\ref] the three TEST totals of Tab.~\ref{}.
      L1608 [Tab.~\ref] \textbf{At three seeds (0 test runs).} Tab.~\ref{} prints
      L1609 [Tab.~\ref] every $K$ at $n=3$ and supersedes the $n=1$ middle rows of Tab.~\ref{}.
      L1663 [Tab.~\ref] Tab.~\ref{}.}{fig:crossover}
      L1672 [Tab.~\ref] two Shapes3D implementations are in Tab.~\ref{}; environments never share a table.}
      L1691 [Tab.~\ref] (Tab.~\ref{}).}
      L1760 [Tab.~\ref] Tab.~\ref{}; the two are never merged.}
      L1799 [Tab.~\ref] punches the hole, as in Tab.~\ref{}. (a)~The n-CLU's content against the content refit without the rows (largest
      L1825 [Tab.~\ref] Tab.~\ref{}.}
      L1849 [Tab.~\ref] selected schedule, so the MLP (seeds 0, 1) and continuous-MLP convergence flags of Tab.~\ref{} travel with them.}
      L1920 [Tab.~\ref] \figslot{fig_encoder_adapt}{\textbf{Encoder adaptation margins.} Single design minus the best learned baseline per arm and seed at $\Kstar$ (Tab.~\ref{}).}{fig:adapt}
      L1929 [Tab.~\ref] declared test run.} A separate instrument; never merged with Tab.~\ref{}. Distinct valid novel
      L1967 [Tab.~\ref] (Tab.~\ref{}). Distinct valid novel tuples per held-out combination (ceiling 36 on row 3, 72 on
      L2049 [Tab.~\ref] Tab.~\ref{}. No n-CLU.}
      L2141 [Tab.~\ref] (Tab.~\ref{}); the before/after contrast is stated in prose (Sec.~\ref{}).}
      L2165 [Tab.~\ref] audited in its own table (Tab.~\ref{}).}
      L2216 [Tab.~\ref] \paragraph{Exact deletion on CMAZE-2b in full.} Tab.~\ref{} is the CMAZE-2b
      L2217 [Tab.~\ref] replication of Sec.~\ref{}: its own table, never merged with Tab.~\ref{}, and
      L2555 [Tabs.~\ref] Tabs.~\ref{} and~\ref{} uses the derangement. The harness smoke numbers above
      L2567 [Tab.~\ref] (Sec.~\ref{}); the symmetric-input rows are Tab.~\ref{}, where a matched
      L2586 [Tab.~\ref] unseen $n$-way & 3 & \textbf{n-CLU row under asymmetric inputs only} (the margin does not survive symmetric inputs, Tab.~\ref{}): n-CLU 0.7246 (0.800 / 0.566 / 0.808) against the best matched baseline
      L2620 [Tab.~\ref] Tab.~\ref{}, whose baselines read the observation alone, were not persisted:
      L2645 [Tab.~\ref] the arms of Tab.~\ref{}:
      L2657 [Tab.~\ref] arms of Tab.~\ref{}:
      L2670 [Tab.~\ref] (across-wells a derangement) and floors are those of Tab.~\ref{}. Values are per seed; the
      L2676 [Tab.~\ref] Tab.~\ref{} and no CMAZE-O, CMAZE-1, CMAZE-2 or CMAZE-2b row.}
      L2718 [Tab.~\ref] withdrawn. Byte convention as Tab.~\ref{}. Values are mean (seed 0 / 1 / 2). This table
      L2766 [Tabs.~\ref] training-steps objection. Tabs.~\ref{} and~\ref{} are unaffected.
      L2788 [Tab.~\ref] The comparator here is a 180k-parameter MLP, a different opponent from Tab.~\ref{}, so
      L2808 [Tab.~\ref] longer, full-decoder leg (Tab.~\ref{}, a block of its own, never merged with the
      L2809 [Tab.~\ref] shortened-decoder leg of Tab.~\ref{}) the ordering is \textbf{not reproduced} by its
      L2886 [Tab.~\ref] the row: the per-arm generated states of the arms of Tab.~\ref{} in
      L2888 [Tab.~\ref] permanently blocked, not pending: the run behind Tab.~\ref{} persisted no arm object and
      L2892 [Tabs.~\ref] (Tabs.~\ref{} and~\ref{}); the n-CLU-content ladder baseline set, fitted on each rung's
      L2893 [Tab.~\ref] 32,768 content rows (Tab.~\ref{}); the four deletion measurements on every scope
      L2894 [Tab.~\ref] (Tab.~\ref{}); the CMAZE-T symmetric-input conditions at six seeds
      L2895 [Tab.~\ref] (Tab.~\ref{}); LACE and PoE with the generation-table run (Sec.~\ref{},
      L2896 [Tabs.~\ref] Tabs.~\ref{} and~\ref{}); the decomposition paragraph
      L2897 [Tab.~\ref] (App.~\ref{}, Tab.~\ref{}); and the coverage control
      L2902 [Tab.~\ref] Tab.~\ref{}.
      L2923 [Tab.~\ref] \item The per-arm generated states of the CMAZE-T asymmetric-input arms (Tab.~\ref{}),
App.~                    31
      L408 [App.~\ref] 12,203 B and 30,507 B; App.~\ref{}, Tab.~\ref{}).
      L410 [App.~\ref] App.~\ref{}, labelled as the endpoint, with its sign stated as a property of $K=5$.
      L485 [App.~\ref] matched. Baseline pins, geometry and the stress comparison are in App.~\ref{}.
      L789 [App.~\ref] and~\ref{}, App.~\ref{}). Each retrain reads 120,028 to 129,761
      L802 [App.~\ref] run (App.~\ref{}).
      L873 [App.~\ref] $\Kstar$ (a declared not-run, App.~\ref{}). \TODO{no record prints the byte ratio of the composition-only n-CLU to the single design;
      L888 [App.~\ref] Tab.~\ref{} (App.~\ref{}) prints both ablations per seed. The pair shows
      L900 [Apps.~\ref] and ``unfreezing hurts'' does not survive the longer leg (Apps.~\ref{}
      L908 [App.~\ref] Tab.~\ref{} (App.~\ref{}). \emph{This is LACE and PoE as instantiated in
      L917 [App.~\ref] App.~\ref{}). All four landscape controls collapse on every seed (declared criterion S1,
      L986 [App.~\ref] App.~\ref{}).
      L1008 [App.~\ref] oracle stays at 1.0000 (Tabs.~\ref{} and~\ref{}, App.~\ref{}).
      L1011 [App.~\ref] its side effects are declared (App.~\ref{}).
      L1107 [App.~\ref] and~\ref{}, App.~\ref{}).
      L1205 [App.~\ref] and every negative with its link are in App.~\ref{}.
      L1219 [App.~\ref] 1.0000 on the seen row), so no row is claimed for the n-CLU (App.~\ref{}). On
      L1246 [App.~\ref] (App.~\ref{}).
      L1265 [App.~\ref] exhausted placement dial, and comparators not run) are listed in App.~\ref{}.
      L1278 [App.~\ref] (derivation in App.~\ref{}). We asserted this in code: the zero-step arm was required to be bit-identical to
      L1289 [App.~\ref] The identity holds under four named conditions (App.~\ref{}).
      L1298 [App.~\ref] environment's dynamics; here the dynamics are the read, not a model of the data.) Its laws are this paper's own theory, each with a status label and rider (further laws in App.~\ref{}).
      L1315 [App.~\ref] App.~\ref{}.
      L1339 [App.~\ref] ablation. What it must contain is fixed by this study's negatives (App.~\ref{}).
      L1360 [Apps.~\ref] Apps.~\ref{} and~\ref{}. An anonymised code archive will accompany the
      L1401 [App.~\ref] three seeds inside one implementation (App.~\ref{}).
      L1414 [App.~\ref] Per-seed geometry and the selection rule's val/test check are in App.~\ref{}.
      L1550 [App.~\ref] App.~\ref{}: on its own full-square convention it is 23.886 to 24.875 times the
      L1876 [App.~\ref] against these baselines (App.~\ref{}).
      L2796 [App.~\ref] different instrument, reported in its own table, App.~\ref{}) asked whether
      L2878 [App.~\ref] (App.~\ref{}).
      L2897 [App.~\ref] (App.~\ref{}, Tab.~\ref{}); and the coverage control
Eq.~                      0
Alg.~                     0
bare \ref                 7
      L134 [\ref{] (\S\ref{}, Table~\ref{}) replaces the earlier asymmetric-input sentence; no
      L263 [\ref{] (Secs.~\ref{}--\ref{}).
      L301 [\ref{] renderer of the result panels (Figs.~\ref{}--\ref{}), so
      L1106 [\ref{] instruments never share a table (Tabs.~\ref{}, \ref{}
      L2887 [\ref{] Figs.~\ref{}, \ref{} and~\ref{}. They are
      L2899 [\ref{] Figs.~\ref{}, \ref{}, \ref{}
      L2899 [\ref{] Figs.~\ref{}, \ref{}, \ref{}
```

### DRAFT-v2-style

```
rival                     0
store                     0
gate                      0
artifact/artefact         0
pre-register              0
A1                        0
A10                       0
C-512                     0
improved                  0
config                   26
      L648 [configuration] measured} (a new TEST configuration would be a new test run, and the \texttt{composition} target is
      L1080 [configuration] compressed baseline. At this configuration the n-CLU's read size holds more latents than the split
      L1158 [configuration] relational baselines is sizeable only on seed 2 (+0.165). We report the configuration as evidence about
      L1182 [configuration] validation $n$-way against the n-CLU's 0.640 / 0.730 / 0.730, so no CMAZE-O configuration can pass
      L1188 [configurations] other two CMAZE-O configurations fail validation with their own links (the autoencoder latent at
      L1197 [configuration] n-CLU is ahead on three of six seeds and its relational configuration is a request-coding result, and
      L1198 [configuration] the CMAZE-O control check with its withdrawn margin. The run count, the configuration ladder, the fallback
      L1231 [configuration] $\Kstar$ is not the point of smallest footprint; and on the one configuration at $K\ge36$ whose composition
      L1314 [configuration] baselines is small and seed-dependent, and its best configuration is a request coding any learner can be
      L1387 [configurations] \paragraph{Seeds and configurations.} Seed count and configuration count are separate fields. The
      L1387 [configuration] \paragraph{Seeds and configurations.} Seed count and configuration count are separate fields. The
      L1388 [configurations] phrase ``nine seeds'' never applies: wherever nine legs appear, they are three configurations
      L1665 [configurations] implementation & seeds & configurations & test runs & band \\
      L1677 [configurations] shared.} Seeds and configurations are separate fields; the nine legs of row 3 are 3 configurations
      L1677 [configurations] shared.} Seeds and configurations are separate fields; the nine legs of row 3 are 3 configurations
      L1684 [configurations] implementation & seeds & configurations & test runs & band \\
      L1874 [configurations] \caption{\textbf{Encoder adaptation at $\Kstar$, TEST, 3 seeds $\times$ 3 configurations (nine
      L2087 [configurations] per row, (b), (c)~six seeds and two configurations.} (a)~Validity under the engine's judge at each seed for the n-CLU (circles) and for the
      L2577 [configurations] unseen $n$-way & 3 & \textbf{n-CLU row under asymmetric inputs only} (the margin does not survive symmetric inputs, Tab.~\ref{}): n-CLU 0.7246 (0.800 / 0.566 / 0.808) against the best matched baseline
      L2629 [configuration] steps), a diagnostic and not a matched baseline. Every panel is re-read from this configuration's
      L2733 [Configuration] \paragraph{Configuration ladder and run count.} Every configuration was measured on validation first,
      L2733 [configuration] \paragraph{Configuration ladder and run count.} Every configuration was measured on validation first,
      L2735 [configurations] 53 configurations on three seeds left 25/53 eligible under the control check, and the crossover rule picked
      L2749 [configuration] configuration passes validation against converged baselines; a field configuration would need $n$-way
      L2749 [configuration] configuration passes validation against converged baselines; a field configuration would need $n$-way
      L2750 [configuration] of at least 1.02 to clear the 0.99 MLP, and seeding a configuration that fails validation is
spoke                     0
hub                       0
wave                      0
charter                   0
ledger                    0
bar k                     0
scored run                0
budget                    0
spend/spent               0
price/priced/pricing      0
buy                       0
pay                       0
cost                      8
      L243 [cost] \item \textbf{Exact deletion on the n-CLU and on its address book, with its cost and its limitation
      L734 [cost] \multicolumn{8}{l}{\textbf{cost}, seconds per seed (0 / 1 / 2)} \\
      L811 [cost] $\Kstar$ object, two holes cost 2 bytes, read 0 data rows, take 0.0000 s and refit nothing, in
      L824 [cost] \caption{\textbf{Refusal DISPLAY rows. A display is never a result.} Our side is a cost line
      L853 [costs] 4,096 data rows read, 521,916 bytes rewritten and 10.48 s, whereas the coupling hole costs one
      L886 [cost] cost printed in Tab.~\ref{}.
      L1316 [cost] memory and time cost of exactness, in bytes and in the book's refit. Switched on, the dynamics lower composition at
      L1811 [cost] \caption{\textbf{Deletion cost for the data-matched baselines: retrain without the rows, on the n-CLU's
cheap                     1
      L1225 [cheap] \item \textbf{Exact deletion is not cheap in memory or time.} The book's exact statistic is 0.97 GB,
expensive                 0
t_pay                     0
corner                    0
held-out composition      0
oblong                    0
exact judge               0
exact                   114
      L81 [Exactly] \title{naive-CLU: Exactly Deletable Compositional Generation\\ on a Frozen Encoder}
      L96 [exactly] training rows exactly. Compositional generation is extensively studied and exact unlearning established. We work at the level of a single \emph{naive unit} and
      L96 [exact] training rows exactly. Compositional generation is extensively studied and exact unlearning established. We work at the level of a single \emph{naive unit} and
      L108 [exactly] controls collapsing to exactly 0.0000.
      L120 [exact] Deletion is exact on the n-CLU and on its address book: after a value, a conjunction or a random
      L122 [exact] most \sci{5.38}{-07}, book statistic deviation 0). It is not free: the book's exact statistic is
      L139 [exactly] dynamical successor: the n-CLU is exactly the zero-dynamics limit of a settling unit, its zero-step arm bit-identical to the static read.
      L167 [Exact] Exact unlearning is likewise established: for $k$-means (Ginart et al., 2019\nocite{}), for bilinear
      L181 [exact] false positives and none exact; no work among the 3,038 combines frozen-decoder, tensor-head
      L186 [exact] \emph{generative} $\wedge$ \emph{exact deletion}, returns zero works on the screen.
      L193 [exactly] exactly unlearnable by splitting its data into shards and retraining the unit on the shard that
      L195 [exact] whose only exact route is the retrain of Sec.~\ref{}.
      L209 [exact] tends to instability, and give no rigorous error analysis. Our route is exact for the opposite
      L224 [Exact] \item \textbf{Exact deletion}: after a set of training rows is deleted, the deletable object
      L227 [exactly] However, the three pull against each other: a lookup table deletes exactly but
      L228 [exactly] composes nothing it has not written; a learned generator composes but can delete exactly only by
      L229 [exact] retraining (the whole model, or a shard of it~\citep{}); and approximate unlearning~\citep{} is approximate by construction, not exact (Sec.~\ref{}). The object we study therefore makes them propertie
      L243 [Exact] \item \textbf{Exact deletion on the n-CLU and on its address book, with its cost and its limitation
      L248 [exact] training row, but the book's exact statistic is 0.97 GB, the size of its fit data, and its refit
      L251 [exactly] \item \textbf{The same properties on a maze.} Deletion transfers exactly; composition on routes is
      L333 [exact] \paragraph{The retained statistic and exact deletion.} Beside the content the n-CLU keeps, per
      L336 [exact] statistic. The address book keeps its own exact statistic, per request tuple the count and first
      L339 [exactness] (Sec.~\ref{}). We use one definition of exactness throughout, and state its scope
      L343 [Exact] \textsc{Exact is scoped to the n-CLU and its address book.} After deleting a set of rows, the n-CLU's
      L418 [exactly] all four landscape controls read exactly 0.000 on both rows on all three seeds, with the
      L421 [exact] Every baseline is read-path matched (130,433 to 131,891 parameters) and fitted on the n-CLU's exact
      L529 [exactly] rungs $n\ge3$ are exactly constant by construction.
      L531 [exactly] 13 of 15 read exactly 0.0000 and two ($n=3$, seeds 0 and 1) read 0.0083 on
      L545 [exactly] families fitted on the rung's first 32,768 rows, exactly the n-CLU's content rows, at the selected
      L586 [exactly] \multicolumn{7}{l}{\emph{n-CLU-content set: the rung's first 32,768 rows, exactly the n-CLU's content rows; 5,120 steps, ten times 51,200}} \\
      L611 [exactly] $r=-0.96$ to $-0.98$, and every arm not fitted on it is exactly constant.
      L626 [exactly] 32,768 content rows, favour the baselines. They do not: fitted on exactly the n-CLU's content rows,
      L668 [exact] rung's own 131,072-row pool, the n-CLU's exact rows, at 20,480 steps, or at 204,800 where the
      L679 [Exact] \subsection{Exact deletion on the single design: value, conjunction, item}
      L744 [exact] \noindent Measurements (a) and (b) are exact on every scope and seed, and (d)'s check is exact too:
      L744 [exact] \noindent Measurements (a) and (b) are exact on every scope and seed, and (d)'s check is exact too:
      L747 [exact] book's refit reads no pool row but reads its exact statistic, 967,304,800 B (0.97 GB). The book sees
      L766 [exactness] \paragraph{The probe that bounds the claim.} The n-CLU-scoping of exactness is measurable, not
      L785 [exact] A learned baseline's only exact deletion route is to retrain from scratch without the rows. We ran
      L795 [exact] collateral damage the baselines' crude retrain is cleaner than the n-CLU's exact deletion. Nor is the
      L798 [exactness] is exactness by construction and no pool row read. Two caveats travel with these rows: the baselines
      L809 [exact] Refusal is a second exact mechanism on the same object: a coupling table with \textbf{holes}.
      L855 [exact] \noindent The hole, not deletion in general, is the operation that is both exact on bytes and
      L858 [exactly] while punching a hole refuses at 1.0000 with a change of exactly $+0.0000$.
      L859 [exactness] The claim is exactness, never erasure without loss.
      L865 [exact] sufficient statistic, which is the requirement it does not meet and the reason it has no exact
      L901 [exact] on rows 3--4, render no exact-match request on the generation table on any seed, and there score at
      L943 [exact] We ran the n-CLU's exact deletion inside the three declared test runs on CMAZE-2: the
      L953 [Exact] \caption{\textbf{Exact deletion on the maze, CMAZE-2, TEST, 3 of 3 scored seeds.} Delete the
      L974 [Exactness] \noindent Exactness is not free here either. As declared in advance, composition on the surviving
      L988 [exactness] three seeds, so the claim is n-CLU-scoped exactness and nothing more.
      L996 [exactly] the 1,449 optimal routes, 1,242 picked up exactly one key, so there was no key order for the
      L1084 [exactly] from other mazes' routes, the published tripwire, reads exactly 0.0000 at $r=1$ and $r=2$ on
      L1127 [exact] control stays below the n-CLU on every seed (highest 0.495, seed 5), and deletion stays exact on all
      L1169 [Exact] printed in Tab.~\ref{} and is not a comparison of like with like. Exact deletion holds on
      L1172 [exactly] n-CLU reads exactly as the n-CLU fitted without the rows, in 0.09 to 0.88 s with no gradient step,
      L1190 [Exact] no single failing link. Exact deletion also runs on the field design (relative change of the statistic 0.0 on
      L1220 [exact] is exact on the n-CLU and on the address book, but deleting the red sphere lowers composition on
      L1225 [Exact] \item \textbf{Exact deletion is not cheap in memory or time.} The book's exact statistic is 0.97 GB,
      L1225 [exact] \item \textbf{Exact deletion is not cheap in memory or time.} The book's exact statistic is 0.97 GB,
      L1228 [Exact] \item \textbf{Exact requires a large number of bytes.} The deletable object at $\Kstar$ is 4.0768 to 4.0897
      L1243 [exact] in this feature space, render no exact-match request on the generation table and score at the level
      L1277 [exactly] $z_{\text{req}}$ exactly at $\lambda=0$. Switching the dynamics on lowers composition at every
      L1310 [exact] The naive unit provides exact deletion by subtraction and, at a stated partition, composition on
      L1316 [exactness] memory and time cost of exactness, in bytes and in the book's refit. Switched on, the dynamics lower composition at
      L1338 [Exact] involves no human subjects and no personal data. Exact deletion is relevant to data-protection
      L1339 [exactness] requests; we claim exactness on the n-CLU and its address book only and no certified unlearning, and
      L1340 [exact] we state in Sec.~\ref{} a case in which exact deletion of a conjunction changes what the
      L1445 [exact] (0.1204), and neither renders an exact-match request on any seed.}
      L1489 [exact] exact-match requests on the generation table as well; in the words of that experiment's own record, a
      L1654 [Exact] \section{Exact deletion: three implementations, and the failure they agree on}
      L1659 [Exact] \caption{\textbf{Exact (n-CLU-scoped) deletion, implementation 1: the maze n-CLU (CMAZE-2).} The
      L1668 [exactly] 1. maze n-CLU (CMAZE-2, TEST) & 3 & 1 & 3 & \sci{7.522}{-16} .. \sci{8.826}{-16} on the statistic; projector exactly 0.000 \\
      L1676 [Exact] \caption{\textbf{Exact (n-CLU-scoped) deletion on Shapes3D, implementations 2 and 3, bands not
      L1699 [exactness] for an exactness claim. Over the pinned deletion only, the narrower field is \sci{2.365}{-14} /
      L1712 [exactly] and the statistic alone is 23.5121 times; a residual of exactly 16,384 B between the serialised
      L1715 [exactness] \paragraph{The failure agreement.} The exactness assertion first failed on TEST seed 2, on the
      L1728 [exactness] fifth (rank over width 0.31 to 0.21). It is the only place where the exactness claim removes content.
      L1733 [exactness] reading 120,048 rows, exactness \sci{1.455}{-11}, survivor $\Delta$ $+0.1250$, $\cos_{\min}$
      L1734 [exactness] 0.183704. Red sphere: 5,507 rows, 0.6205 s, reference 10.843 s / 125,565 rows, exactness
      L1736 [exactness] 0.2554 s, reference 11.446 s / 129,761 rows, exactness \sci{1.455}{-11}, survivor $\Delta$
      L1789 [exactly] of the rendered features; the book's count and first-moment statistic deviate by exactly 0), per
      L1898 [exact] \noindent The factor-probe margins over random initialisation (aggregate / exact-match, per seed,
      L2107 [exactly] coding, not from the n-CLU. On seed 4 the random-atoms control also equals the n-CLU exactly
      L2204 [Exact] \paragraph{Exact deletion on CMAZE-2b in full.} Tab.~\ref{} is the CMAZE-2b
      L2211 [Exact] checkout reproduces both. Exact is not light in extent: the exact-head downdate changes every atom
      L2211 [exact] checkout reproduces both. Exact is not light in extent: the exact-head downdate changes every atom
      L2216 [exactness] 0.4404, 0.5638 $\to$ 0.5319), so the claim is n-CLU-scoped exactness and nothing more. Speed is not
      L2218 [exact] content and head from the survivors. The difference is that it is exact without keeping the items,
      L2226 [Exact] \caption{\textbf{Exact deletion on the maze, CMAZE-2b, 3 seeds, 0 test runs.} This table
      L2242 [exact] exact-head downdate: seconds & 3 & 2.042 / 2.328 / 2.181 \\
      L2291 [exactly] the span: pushed off the span by $\delta=0.10$, the settle moves by exactly $\delta/(1+\beta)$,
      L2295 [exactly] of the span form. (C4) There is no cross-well term: at $\lambda=0$ the displacement is exactly 0.0,
      L2317 [exact] the exact log-derivative (numeric $+1.138526$ against $+1.138500$). The underdamped slope is
      L2320 [exactly] branch the slow multiplier is exactly $\sqrt{1-\gamma}$, so a measured floor slope has one degree of
      L2322 [exactly] residual field in the records (\texttt{d14\_confinement}, exactly 0.0 on 60 cells by construction)
      L2347 [exact] B4 & 3 & \ldots against the exact inversion & 0.3782 & 0.3784 & 0.3834 \\
      L2369 [exact] half-life form is already 2.4\% off at $h=0.30h^\star$ and is never quoted; only the exact root is.
      L2371 [exactly] The seed-0 leg replicates bit-exactly from a clean checkout, and two of the three declared runs
      L2393 [exactly] against composition. An 8-byte rule ties the field-mode n-CLU exactly on the two symbolic refusal rows,
      L2411 [exact] \paragraph{Memory-level exact deletion in a designed memory (\textsc{confirmed [prior
      L2412 [exact] derivation]}).} With an order-independent waitlist, memory-level deletion is exact at every load
      L2424 [exact] rollout prediction on this n-CLU).} On the exact one-step Jacobian, the emergent retention
      L2435 [exactly] basin displacement is exactly zero off the coupling graph (0 of 4656 off-graph entries nonzero
      L2478 [exact] memory of 6.3 KB against 6.00 MB at $T=1024$; the match is exact only at zero damping.
      L2483 [exact] read size, an n-CLU cannot simultaneously have exact value fidelity, amplitude-independent
      L2500 [exactly] 39\% exactly optimal).
      L2503 [exact] \item Run a matched-bytes stitching tripwire with a fragment-length sweep before adoption: exact-key
      L2583 [exactness] deletion, value & 3 & colour 1; n-CLU-scoped exactness $S_k$ relative change \sci{7.80}{-16} / \sci{1.22}{-15} / \sci{1.09}{-15}, atoms change 0; downdated read equal to fitted-without: 0.4369 / 0.425
      L2630 [exactly] persisted objects; the re-read reproduces all 132 scored TEST arm scores exactly, over seeds 0--5.
      L2693 [exactness] deletion & 6 & base n-CLU, words + observation: n-CLU-scoped exactness on every seed (relative statistic change at most \sci{1.94}{-15}, atoms unchanged, reads after the downdate identical to the n-CL
      L2725 [exactness] deletion, conj. & 3 & 2240 / 1435 / 2168 rows deleted; relative change 0.0; deleted side 0.5800 / 0.8000 / 0.7067 $\to$ 0.0000; other colours 0.6457 $\to$ 0.6343, 0.6880 $\to$ 0.6800, 0.7160 $\to$ 0.7
      L2824 [exact] On an exact quadratic, where the law holds by construction, the same measurement chain recovers it
      L2850 [exact] law. Two consequences derived before measurement were then exact: the refusal horizon is $19.9316$
10x                       0
landscape_removed         0
landscape_shuffled        0
random_atoms              0
Naive CLU                 0
CLU (bare)                3
      L1269 [CLU] The n-CLU we evaluate is the zero-dynamics limit of the CLU: the same content, wells and read,
      L1323 [CLU] \section{Follow-up: a full evaluation of the physics CLU}
      L1327 [CLU] The leaving note states what the naive unit does not provide; the follow-up is therefore a full evaluation of the settling (physics) CLU on the three properties of Sec.~\ref{}, with the same judges, m
\S                        1
      L136 [\S] (\S\ref{}, Table~\ref{}) replaces the earlier asymmetric-input sentence; no
Section~                  0
Figure~                   0
Table~                    1
      L136 [Table~\ref] (\S\ref{}, Table~\ref{}) replaces the earlier asymmetric-input sentence; no
Appendix~                 0
Sec.~                    42
      L153 [Sec.~\ref] (Sec.~\ref{}). We claim neither the best compositional-generation system nor the best
      L190 [Sec.~\ref] than about three months is incomplete (Sec.~\ref{}).
      L195 [Sec.~\ref] whose only exact route is the retrain of Sec.~\ref{}.
      L229 [Sec.~\ref] retraining (the whole model, or a shard of it~\citep{}); and approximate unlearning~\citep{} is approximate by construction, not exact (Sec.~\ref{}). The object we study therefore makes them propertie
      L240 [Sec.~\ref] to +21.936 over the transformer (Sec.~\ref{}). On every rung of an $n$-way ladder it
      L242 [Sec.~\ref] mean (Sec.~\ref{}); for $n\ge3$ the rungs are a training-data axis and not a depth axis.
      L247 [Sec.~\ref] about the value it shares with the survivors (Sec.~\ref{}). The deletion reads no
      L250 [Sec.~\ref] separate, structural mechanism, a hole in a coupling table (Sec.~\ref{}).
      L253 [Sec.~\ref] matched-bytes lookup floor more than it lifts the n-CLU (Sec.~\ref{}). On a transition
      L258 [Sec.~\ref] (Sec.~\ref{}). We offer the transition instrument itself as a minor contribution.
      L260 [Sec.~\ref] forget, while an item memory forgets but cannot compose (Sec.~\ref{}). The
      L262 [Secs.~\ref] close the paper (Secs.~\ref{}--\ref{}).
      L272 [Sec.~\ref] (Sec.~\ref{}). The rows still pending a run are marked placeholders whose text names the row.
      L339 [Sec.~\ref] (Sec.~\ref{}). We use one definition of exactness throughout, and state its scope
      L395 [Sec.~\ref] Sec.~\ref{} on the Shapes3D value hold-out~\citep{}, with the partition fixed on validation
      L487 [Sec.~\ref] \paragraph{A second, earlier baseline set.} The ladder (Sec.~\ref{}) also prints an earlier set
      L543 [Sec.~\ref] combination-split pool that carries the held-out values (Sec.~\ref{}), at 5,040 optimiser
      L689 [Sec.~\ref] hole for $c$ (Sec.~\ref{}). Every deletion row therefore prints four measurements
      L780 [Sec.~\ref] is smallest. This is why refusal needs its own mechanism (Sec.~\ref{}).
      L905 [Sec.~\ref] the study has no functioning generative baseline (Sec.~\ref{}).
      L925 [Sec.~\ref] Sec.~\ref{} tests the three properties on images; a route task with the engine's integer judge is, however, the harder test: one wrong action is a zero. We therefore use a colour maze whose engine, so
      L937 [Sec.~\ref] \item \textbf{CMAZE-T} and \textbf{CMAZE-O}: transition and observation instruments, reported in Sec.~\ref{}.
      L947 [Sec.~\ref] world-model instruments of Sec.~\ref{} also do), and it replicates from a clean
      L981 [Sec.~\ref] The replication on CMAZE-2b, the environment of the verb result (Sec.~\ref{}), is its own
      L1095 [Sec.~\ref] The two instruments below are the maze counterpart of Sec.~\ref{}: they ask the n-CLU to
      L1215 [Sec.~\ref] CMAZE-2b the controls again fail to collapse (Sec.~\ref{}).
      L1218 [Sec.~\ref] (Sec.~\ref{}, Tab.~\ref{}).
      L1223 [Sec.~\ref] baseline without the same rows is cleaner on collateral damage (Sec.~\ref{},
      L1224 [Sec.~\ref] Sec.~\ref{}).
      L1238 [Sec.~\ref] per held-out combination from ten to thirty times its steps (Sec.~\ref{}).
      L1327 [Sec.~\ref] The leaving note states what the naive unit does not provide; the follow-up is therefore a full evaluation of the settling (physics) CLU on the three properties of Sec.~\ref{}, with the same judges, m
      L1340 [Sec.~\ref] we state in Sec.~\ref{} a case in which exact deletion of a conjunction changes what the
      L1345 [Sec.~\ref] and the provenance of each is kept beside it in the source. Sec.~\ref{} defines the
      L1457 [Sec.~\ref] \multicolumn{5}{p{15cm}}{\emph{LACE and PoE are added beside the baselines above, which are not replaced; each runs at its own declared settings (LACE 31.08$\times$, PoE 27.135$\times$ the 524,288 B r
      L1650 [Sec.~\ref] resolve the validation plateau (Sec.~\ref{}); the values are those of
      L2129 [Sec.~\ref] (Tab.~\ref{}); the before/after contrast is stated in prose (Sec.~\ref{}).}
      L2205 [Sec.~\ref] replication of Sec.~\ref{}: its own table, never merged with Tab.~\ref{}, and
      L2271 [Sec.~\ref] refusal score on a route request. The Shapes3D coupling hole (Sec.~\ref{}), in contrast, is a
      L2419 [Sec.~\ref] paper adopts in Sec.~\ref{}: an n-CLU-level guarantee only, with the frozen encoder and any
      L2558 [Sec.~\ref] (Sec.~\ref{}); the symmetric-input rows are Tab.~\ref{}, where a matched
      L2886 [Sec.~\ref] (Tab.~\ref{}); LACE and PoE with the generation-table run (Sec.~\ref{},
      L2889 [Sec.~\ref] (Sec.~\ref{}). No figure is a placeholder figure and none carries an in-image banner.
Fig.~                    10
      L300 [Figs.~\ref] renderer of the result panels (Figs.~\ref{}--\ref{}), so
      L1107 [Fig.~\ref] (Fig.~\ref{}). Test mazes are generated at random, so a lookup of training mazes
      L2108 [Fig.~\ref] (0.9004, on every held-out tuple, Fig.~\ref{}); on seed 5 the best matched
      L2623 [Fig.~\ref] Fig.~\ref{}.}{fig:cmazet-per-arm}
      L2627 [Fig.~\ref] Fig.~\ref{}, for n-CLU-REL and its arms, and for the six baselines
      L2631 [Fig.~\ref] Under each $s_{t+1}$, its emitted facts, as in Fig.~\ref{}.}{fig:cmazet-rel-per-arm}
      L2635 [Fig.~\ref] Columns, and the emitted facts under each $s_{t+1}$, as in Fig.~\ref{}. For
      L2647 [Fig.~\ref] without; under each $s_{t+1}$, its emitted facts, as in Fig.~\ref{}. For the
      L2878 [Figs.~\ref] Figs.~\ref{}, \ref{} and~\ref{}. They are
      L2890 [Figs.~\ref] Figs.~\ref{}, \ref{}, \ref{}
Tab.~                    86
      L407 [Tab.~\ref] 12,203 B and 30,507 B (App.~\ref{}, Tab.~\ref{}).
      L414 [Tab.~\ref] Tab.~\ref{} is the scored result. Row 3 holds out 60 pair combinations and row 4 holds out
      L449 [Tab.~\ref] Tab.~\ref{}.}
      L483 [Tab.~\ref] object itself is larger, and its memory footprint (Tab.~\ref{}) is printed,
      L525 [Tab.~\ref] $n=6$ are the two ends. Tab.~\ref{} gives the n-CLU at $\Kstar$ on every rung and
      L550 [Tab.~\ref] and PoE are in Tab.~\ref{}. $n=2$: 60 combinations; $n\ge3$: the identical
      L615 [Tab.~\ref] On every rung and every seed the n-CLU's score in Tab.~\ref{} exceeds every baseline's
      L643 [Tab.~\ref] \emph{derived, not measured}: it is the $n=3$ n-CLU cell of Tab.~\ref{}, and no new
      L659 [Tab.~\ref] Tab.~\ref{}: \emph{``We beat the naive fair baselines at every $n$; the n-CLU's own 2-way
      L670 [Tab.~\ref] (Tab.~\ref{}). Open markers are the seeds; the line is their mean. The n-CLU's margin
      L690 [Tab.~\ref] (Tab.~\ref{}): (a) the n-CLU's content equals the content refit without the rows, a result; (b) the book
      L775 [Tab.~\ref] now also downdates the address book's statistics (measurement (b) of Tab.~\ref{}). The trade-off moved
      L787 [Tabs.~\ref] pool, at the baseline's selected schedule, three seeds (Tabs.~\ref{}
      L800 [Tab.~\ref] of Tab.~\ref{} travel with them; and approximate unlearning of the data-matched baselines was not
      L817 [Tab.~\ref] holed value is refused by construction, so refusal is measurement (c) of Tab.~\ref{}, a
      L871 [Tab.~\ref] the records give the two totals printed here and in Tab.~\ref{}, and they are
      L883 [Tab.~\ref] Tab.~\ref{} (App.~\ref{}) prints both ablations per seed. Together they
      L886 [Tab.~\ref] cost printed in Tab.~\ref{}.
      L903 [Tab.~\ref] Tab.~\ref{} (App.~\ref{}). \emph{This is LACE and PoE as instantiated in
      L908 [Tab.~\ref] The generation table is a second Shapes3D instrument, never merged with Tab.~\ref{}: 72
      L911 [Tab.~\ref] nothing not-run; Tab.~\ref{},
      L945 [Tab.~\ref] with the n-CLU refit without those rows. Tab.~\ref{} reports the result. It was the
      L982 [Tab.~\ref] table, never merged with Tab.~\ref{}, and uses no test run (Tab.~\ref{},
      L982 [Tab.~\ref] table, never merged with Tab.~\ref{}, and uses no test run (Tab.~\ref{},
      L1005 [Tabs.~\ref] and the oracle stays at 1.0000 (Tabs.~\ref{} and~\ref{}, App.~\ref{}).
      L1090 [Tab.~\ref] The n-CLU's differentiator on the maze is deletion, which transferred (Tab.~\ref{}); composition on routes belongs to the verb.
      L1100 [Tabs.~\ref] instruments never share a table (Tabs.~\ref{}, \ref{}
      L1117 [Tab.~\ref] Tab.~\ref{} gives every input condition, and within a condition every arm receives the
      L1153 [Tab.~\ref] Tab.~\ref{}) reach 0.915 / 0.869 / 0.819 / 0.921 / 0.900 / 0.939, so the n-CLU's lead
      L1169 [Tab.~\ref] printed in Tab.~\ref{} and is not a comparison of like with like. Exact deletion holds on
      L1218 [Tab.~\ref] (Sec.~\ref{}, Tab.~\ref{}).
      L1229 [Tab.~\ref] times the reference (Tab.~\ref{}); across nine swept partitions the honest
      L1244 [Tab.~\ref] of our landscape controls (Tab.~\ref{}); the study compares naive units only and has no
      L1392 [Tab.~\ref] \section{Tab.~\ref{}: what it does and does not claim}
      L1394 [Tab.~\ref] The baselines of Tab.~\ref{} are matched on the read path at 130,514 (MLP), 130,721 (LSTM),
      L1400 [Tab.~\ref] in Tab.~\ref{} and is never matched. Baselines are matched on read-path
      L1407 [Tab.~\ref] generation-2 cache.} A separate instrument from Tab.~\ref{}, never merged with it:
      L1440 [Tab.~\ref] 2).} Distinct valid novel tuples per held-out combination, on the combinations of Tab.~\ref{} (upper block) and of
      L1441 [Tab.~\ref] every rung of Tab.~\ref{} (lower block). Neither is matched on the read path or on bytes;
      L1443 [Tab.~\ref] the generation table (Tab.~\ref{}) their aggregates, 0.3032 and 0.1968, sit at the
      L1477 [Tab.~\ref] learned baselines on rows 3--4 and on every ladder rung (Tab.~\ref{}), 3 seeds; no
      L1499 [Tab.~\ref] baseline by +3.780 / +6.872 / +3.697 (mean +4.783). On the ladder, Tab.~\ref{} prints the $K=5$
      L1509 [Tab.~\ref] Tab.~\ref{} (each rung's own 131,072 rows) and against the strongest baseline of the earlier
      L1511 [Tab.~\ref] Tab.~\ref{}:} for $n\ge3$ the rungs differ in fit-pool coverage, not in composition
      L1563 [Tab.~\ref] honest four-part total is U-shaped. The $n=1$ middle rows are superseded by Tab.~\ref{},
      L1593 [Tab.~\ref] \emph{validation} pin leg, distinct from the three TEST totals of Tab.~\ref{}.
      L1596 [Tab.~\ref] \textbf{At three seeds (0 test runs).} Tab.~\ref{} prints
      L1597 [Tab.~\ref] every $K$ at $n=3$ and supersedes the $n=1$ middle rows of Tab.~\ref{}.
      L1651 [Tab.~\ref] Tab.~\ref{}.}{fig:crossover}
      L1660 [Tab.~\ref] two Shapes3D implementations are in Tab.~\ref{}; environments never share a table.}
      L1679 [Tab.~\ref] (Tab.~\ref{}).}
      L1748 [Tab.~\ref] Tab.~\ref{}; the two are never merged.}
      L1787 [Tab.~\ref] punches the hole, as in Tab.~\ref{}. (a)~The n-CLU's content against the content refit without the rows (largest
      L1813 [Tab.~\ref] Tab.~\ref{}.}
      L1837 [Tab.~\ref] selected schedule, so the MLP (seeds 0, 1) and continuous-MLP convergence flags of Tab.~\ref{} travel with them.}
      L1908 [Tab.~\ref] \figslot{fig_encoder_adapt}{\textbf{Encoder adaptation margins.} Single design minus the best learned baseline per arm and seed at $\Kstar$ (Tab.~\ref{}).}{fig:adapt}
      L1917 [Tab.~\ref] declared test run.} A separate instrument; never merged with Tab.~\ref{}. Distinct valid novel
      L1955 [Tab.~\ref] (Tab.~\ref{}). Distinct valid novel tuples per held-out combination (ceiling 36 on row 3, 72 on
      L2037 [Tab.~\ref] Tab.~\ref{}. No n-CLU.}
      L2129 [Tab.~\ref] (Tab.~\ref{}); the before/after contrast is stated in prose (Sec.~\ref{}).}
      L2153 [Tab.~\ref] audited in its own table (Tab.~\ref{}).}
      L2204 [Tab.~\ref] \paragraph{Exact deletion on CMAZE-2b in full.} Tab.~\ref{} is the CMAZE-2b
      L2205 [Tab.~\ref] replication of Sec.~\ref{}: its own table, never merged with Tab.~\ref{}, and
      L2546 [Tabs.~\ref] Tabs.~\ref{} and~\ref{} uses the derangement. The harness smoke numbers above
      L2558 [Tab.~\ref] (Sec.~\ref{}); the symmetric-input rows are Tab.~\ref{}, where a matched
      L2577 [Tab.~\ref] unseen $n$-way & 3 & \textbf{n-CLU row under asymmetric inputs only} (the margin does not survive symmetric inputs, Tab.~\ref{}): n-CLU 0.7246 (0.800 / 0.566 / 0.808) against the best matched baseline
      L2611 [Tab.~\ref] Tab.~\ref{}, whose baselines read the observation alone, were not persisted:
      L2636 [Tab.~\ref] the arms of Tab.~\ref{}:
      L2648 [Tab.~\ref] arms of Tab.~\ref{}:
      L2661 [Tab.~\ref] (across-wells a derangement) and floors are those of Tab.~\ref{}. Values are per seed; the
      L2667 [Tab.~\ref] Tab.~\ref{} and no CMAZE-O, CMAZE-1, CMAZE-2 or CMAZE-2b row.}
      L2709 [Tab.~\ref] withdrawn. Byte convention as Tab.~\ref{}. Values are mean (seed 0 / 1 / 2). This table
      L2757 [Tabs.~\ref] training-steps objection. Tabs.~\ref{} and~\ref{} are unaffected.
      L2780 [Tab.~\ref] Tab.~\ref{}, so the two tables are never merged. A factor probe measured as a margin over
      L2799 [Tab.~\ref] longer, full-decoder leg (Tab.~\ref{}, a block of its own, never merged with the
      L2800 [Tab.~\ref] shortened-decoder leg of Tab.~\ref{}) the ordering is \textbf{not reproduced} by its
      L2877 [Tab.~\ref] the row: the per-arm generated states of the arms of Tab.~\ref{} in
      L2879 [Tab.~\ref] permanently blocked, not pending: the run behind Tab.~\ref{} persisted no arm object and
      L2883 [Tabs.~\ref] (Tabs.~\ref{} and~\ref{}); the n-CLU-content ladder baseline set, fitted on each rung's
      L2884 [Tab.~\ref] 32,768 content rows (Tab.~\ref{}); the four deletion measurements on every scope
      L2885 [Tab.~\ref] (Tab.~\ref{}); the CMAZE-T symmetric-input conditions at six seeds
      L2886 [Tab.~\ref] (Tab.~\ref{}); LACE and PoE with the generation-table run (Sec.~\ref{},
      L2887 [Tabs.~\ref] Tabs.~\ref{} and~\ref{}); the decomposition paragraph
      L2888 [Tab.~\ref] (App.~\ref{}, Tab.~\ref{}); and the coverage control
      L2893 [Tab.~\ref] Tab.~\ref{}.
      L2914 [Tab.~\ref] \item The per-arm generated states of the CMAZE-T asymmetric-input arms (Tab.~\ref{}),
App.~                    31
      L407 [App.~\ref] 12,203 B and 30,507 B (App.~\ref{}, Tab.~\ref{}).
      L409 [App.~\ref] App.~\ref{}, labelled as the endpoint, with its sign stated as a property of $K=5$.
      L484 [App.~\ref] never matched. App.~\ref{} gives the baseline pins, the geometry and the stress
      L788 [App.~\ref] and~\ref{}, App.~\ref{}). Each retrain reads 120,028 to 129,761
      L801 [App.~\ref] run (App.~\ref{}).
      L870 [App.~\ref] $\Kstar$ (a declared not-run, App.~\ref{}). \TODO{no record prints the byte ratio of the composition-only n-CLU to the single design;
      L883 [App.~\ref] Tab.~\ref{} (App.~\ref{}) prints both ablations per seed. Together they
      L895 [Apps.~\ref] and ``unfreezing hurts'' does not survive the longer leg (Apps.~\ref{}
      L903 [App.~\ref] Tab.~\ref{} (App.~\ref{}). \emph{This is LACE and PoE as instantiated in
      L912 [App.~\ref] App.~\ref{}). All four landscape controls collapse on every seed (declared criterion S1,
      L983 [App.~\ref] App.~\ref{}).
      L1005 [App.~\ref] and the oracle stays at 1.0000 (Tabs.~\ref{} and~\ref{}, App.~\ref{}).
      L1008 [App.~\ref] its side effects are declared (App.~\ref{}).
      L1101 [App.~\ref] and~\ref{}, App.~\ref{}).
      L1199 [App.~\ref] and every negative with its link are in App.~\ref{}.
      L1214 [App.~\ref] 1.0000 on the seen row), so no row is claimed for the n-CLU (App.~\ref{}). On
      L1241 [App.~\ref] (App.~\ref{}).
      L1260 [App.~\ref] exhausted placement dial, and comparators not run) are listed in App.~\ref{}.
      L1271 [App.~\ref] to go (derivation in App.~\ref{}). We asserted this in code: the zero-step arm was required to be bit-identical to
      L1281 [App.~\ref] The identity holds under four named conditions (App.~\ref{}).
      L1291 [App.~\ref] App.~\ref{}).
      L1305 [App.~\ref] beat no model, and we make no capability claim from them}; no arm of ours fails the retention law, so it distinguishes nothing. Full statements, criteria and misses: App.~\ref{}.
      L1327 [App.~\ref] The leaving note states what the naive unit does not provide; the follow-up is therefore a full evaluation of the settling (physics) CLU on the three properties of Sec.~\ref{}, with the same judges, m
      L1347 [Apps.~\ref] Apps.~\ref{} and~\ref{} give the per-cell provenance conventions and
      L1389 [App.~\ref] times three seeds inside one implementation (App.~\ref{}).
      L1402 [App.~\ref] Per-seed geometry and the selection rule's val/test check are in App.~\ref{}.
      L1538 [App.~\ref] App.~\ref{}: on its own full-square convention it is 23.886 to 24.875 times the
      L1864 [App.~\ref] against these baselines (App.~\ref{}).
      L2787 [App.~\ref] different instrument, reported in its own table, App.~\ref{}) asked whether
      L2869 [App.~\ref] (App.~\ref{}).
      L2888 [App.~\ref] (App.~\ref{}, Tab.~\ref{}); and the coverage control
Eq.~                      0
Alg.~                     0
bare \ref                 7
      L136 [\ref{] (\S\ref{}, Table~\ref{}) replaces the earlier asymmetric-input sentence; no
      L262 [\ref{] close the paper (Secs.~\ref{}--\ref{}).
      L300 [\ref{] renderer of the result panels (Figs.~\ref{}--\ref{}), so
      L1100 [\ref{] instruments never share a table (Tabs.~\ref{}, \ref{}
      L2878 [\ref{] Figs.~\ref{}, \ref{} and~\ref{}. They are
      L2890 [\ref{] Figs.~\ref{}, \ref{}, \ref{}
      L2890 [\ref{] Figs.~\ref{}, \ref{}, \ref{}
```

