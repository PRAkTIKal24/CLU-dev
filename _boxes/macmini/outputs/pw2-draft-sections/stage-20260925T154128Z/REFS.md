# Section-label targets, before -> after (rendered number), and every `\ref` to a section label

## DRAFT-v2

| label | before | after | refs (line: context) |
|---|---|---|---|
| `sec:intro` | 1 | 1 | (none) |
| `sec:positioning` | 1.1 | 1.1 | (none) |
| `sec:why` | 1.2 | 1.2 | 1227: …Sec.~\ref{sec:why},… |
| `sec:contrib` | 1.3 | 1.3 | (none) |
| `sec:object` | 2 | 2 | 147: …(Sec.~\ref{sec:object})…<br>396: …Sec.~\ref{sec:object} …<br>1258: …defined in Sec.~\ref{sec:object};…<br>2014: …paper adopts in Sec.~\ref{sec:object}:… |
| `sec:experiments` | — | 3 | (none) |
| `sec:shapes` | 3 | 3.1 | 1565: …(Sec.~\ref{sec:shapes})… |
| `sec:maze` | 4 | 3.5 | (none) |
| `sec:results` | — | 4 | (none) |
| `sec:shapes-t7` | 3.1 | 4.1 | 236: …  to +21.936 over the transformer (Sec.~\ref{sec:shapes-t7})…<br>646: … that carries the held-out values (Sec.~\ref{sec:shapes-t7})…<br>1137: …rom ten to thirty times its steps (Sec.~\ref{sec:shapes-t7})… |
| `sec:ladder` | 3.2 | 4.2 | 238: …  mean (Sec.~\ref{sec:ladder})…<br>424: …earlier baseline set.} The ladder (Sec.~\ref{sec:ladder})…<br>2461: …(Sec.~\ref{sec:ladder})… |
| `sec:shapes-del` | 3.3 | 4.3 | 245: …alue it shares with the survivors (Sec.~\ref{sec:shapes-del})…<br>329: …(Sec.~\ref{sec:shapes-del})…<br>1122: …s is cleaner on collateral damage (Sec.~\ref{sec:shapes-del},…<br>1249: …we state in Sec.~\ref{sec:shapes-del} … |
| `sec:del-comparators` | 3.4 | 4.4 | 190: …te to that match is the retrain of Sec.~\ref{sec:del-comparators}.…<br>223: …construction and does not meet it (Sec.~\ref{sec:del-comparators})…<br>1123: …  Sec.~\ref{sec:del-comparators})…<br>1233: … them includes zero on some seeds (Sec.~\ref{sec:del-comparators})… |
| `sec:hole` | 3.5 | 4.5 | 248: …anism, a hole in a coupling table (Sec.~\ref{sec:hole})…<br>457: …hole for $c$ (Sec.~\ref{sec:hole})…<br>875: …y refusal needs its own mechanism (Sec.~\ref{sec:hole})… |
| `sec:ablations` | 3.6 | 4.6 | 256: …memory forgets and cannot compose (Sec.~\ref{sec:ablations})… |
| `sec:bard` | 3.7 | 4.7 | (none) |
| `sec:adapt` | 3.7 | 4.7 | 1369: …gainst each, on every seed. Scope: Sec.~\ref{sec:adapt}.…<br>2458: …PoE with the generation-table run (Sec.~\ref{sec:adapt},… |
| `sec:maze-wm` | 4 | 4.8 | 129: …(\S\ref{sec:maze-wm},…<br>254: …  (Sec.~\ref{sec:maze-wm})…<br>266: …(Sec.~\ref{sec:maze-wm})…<br>1116: …  (Sec.~\ref{sec:maze-wm},…<br>2169: …(Sec.~\ref{sec:maze-wm})…<br>2223: …Sec.~\ref{sec:maze-wm}.…<br>2238: … $s_{t+1}$ is read as described in Sec.~\ref{sec:maze-wm}.…<br>2248: …$s_{t+1}$ is read as described in Sec.~\ref{sec:maze-wm}.…<br>2259: … $s_{t+1}$ is read as described in Sec.~\ref{sec:maze-wm}.… |
| `sec:limits` | 5 | 5 | 185: …three months is incomplete (Sec.~\ref{sec:limits})…<br>258: …  (Secs.~\ref{sec:limits}-… |
| `sec:limits-list` | 5.1 | 5.1 | 1005: …o functioning generative baseline (Sec.~\ref{sec:limits-list})… |
| `sec:clue` | 5.2 | 5.3 | (none) |
| `sec:theory` | 5.3 | 5.4 | (none) |
| `sec:leaving` | 5.4 | 5.5 | (none) |
| `sec:followup` | 6 | 5.6 | 258: …  (Secs.~\ref{sec:limits}--\ref{sec:followup})… |

## DRAFT-v2-style

| label | before | after | refs (line: context) |
|---|---|---|---|
| `sec:intro` | 1 | 1 | (none) |
| `sec:positioning` | 1.1 | 1.1 | (none) |
| `sec:why` | 1.2 | 1.2 | 1218: …cs) CLU on the three properties of Sec.~\ref{sec:why},… |
| `sec:contrib` | 1.3 | 1.3 | (none) |
| `sec:object` | 2 | 2 | 149: …(Sec.~\ref{sec:object})…<br>396: …Sec.~\ref{sec:object} …<br>1247: …record, not what the number means. Sec.~\ref{sec:object} …<br>2006: …paper adopts in Sec.~\ref{sec:object}:… |
| `sec:experiments` | — | 3 | (none) |
| `sec:shapes` | 3 | 3.1 | 1555: …(Sec.~\ref{sec:shapes})… |
| `sec:maze` | 4 | 3.5 | (none) |
| `sec:results` | — | 4 | (none) |
| `sec:shapes-t7` | 3.1 | 4.1 | 236: …  to +21.936 over the transformer (Sec.~\ref{sec:shapes-t7})…<br>646: … that carries the held-out values (Sec.~\ref{sec:shapes-t7})…<br>1134: …rom ten to thirty times its steps (Sec.~\ref{sec:shapes-t7})… |
| `sec:ladder` | 3.2 | 4.2 | 238: …  mean (Sec.~\ref{sec:ladder})…<br>424: …earlier baseline set.} The ladder (Sec.~\ref{sec:ladder})…<br>2453: …(Sec.~\ref{sec:ladder})… |
| `sec:shapes-del` | 3.3 | 4.3 | 245: …alue it shares with the survivors (Sec.~\ref{sec:shapes-del})…<br>330: …(Sec.~\ref{sec:shapes-del})…<br>1119: …s is cleaner on collateral damage (Sec.~\ref{sec:shapes-del},…<br>1239: …we state in Sec.~\ref{sec:shapes-del} … |
| `sec:del-comparators` | 3.4 | 4.4 | 191: …te to that match is the retrain of Sec.~\ref{sec:del-comparators}.…<br>225: …construction and does not meet it (Sec.~\ref{sec:del-comparators})…<br>1120: …  Sec.~\ref{sec:del-comparators})…<br>1223: … them includes zero on some seeds (Sec.~\ref{sec:del-comparators})… |
| `sec:hole` | 3.5 | 4.5 | 248: …anism, a hole in a coupling table (Sec.~\ref{sec:hole})…<br>457: …hole for $c$ (Sec.~\ref{sec:hole})…<br>875: …y refusal needs its own mechanism (Sec.~\ref{sec:hole})… |
| `sec:ablations` | 3.6 | 4.6 | 256: …memory forgets but cannot compose (Sec.~\ref{sec:ablations})… |
| `sec:bard` | 3.7 | 4.7 | (none) |
| `sec:adapt` | 3.7 | 4.7 | 1359: …gainst each, on every seed. Scope: Sec.~\ref{sec:adapt}.…<br>2450: …PoE with the generation-table run (Sec.~\ref{sec:adapt},… |
| `sec:maze-wm` | 4 | 4.8 | 132: …(\S\ref{sec:maze-wm},…<br>254: …  (Sec.~\ref{sec:maze-wm})…<br>266: …(Sec.~\ref{sec:maze-wm})…<br>1113: …  (Sec.~\ref{sec:maze-wm},…<br>2161: …(Sec.~\ref{sec:maze-wm})…<br>2215: …Sec.~\ref{sec:maze-wm}.…<br>2230: … $s_{t+1}$ is read as described in Sec.~\ref{sec:maze-wm}.…<br>2240: …$s_{t+1}$ is read as described in Sec.~\ref{sec:maze-wm}.…<br>2251: … $s_{t+1}$ is read as described in Sec.~\ref{sec:maze-wm}.… |
| `sec:limits` | 5 | 5 | 186: … about three months is incomplete (Sec.~\ref{sec:limits})…<br>258: …  close the paper (Secs.~\ref{sec:limits}-… |
| `sec:limits-list` | 5.1 | 5.1 | 1000: …o functioning generative baseline (Sec.~\ref{sec:limits-list})… |
| `sec:clue` | 5.2 | 5.3 | (none) |
| `sec:theory` | 5.3 | 5.4 | (none) |
| `sec:leaving` | 5.4 | 5.5 | (none) |
| `sec:followup` | 6 | 5.6 | 258: …lose the paper (Secs.~\ref{sec:limits}--\ref{sec:followup})… |

