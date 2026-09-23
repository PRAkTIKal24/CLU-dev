"""One-off generator: resolves the pass-2 cell spec into a STATIC register (cells_pass2.py).

ROW  = copy the whole markdown table row on that line (outer pipes stripped).
None = find the unique line carrying the explicit value (error if 0 or >1 hits).
The build later re-checks every (value, line) pair against the file independently.
"""
import sys
from pathlib import Path

EV = Path("/Users/batmac/clu/evidence/outputs")
SRC2 = {
    "W16F": "w16-maze-wm/FINDINGS.md",
    "W16T": "w16-maze-wm/TABLES.md",
    "W16J": "w16-maze-wm/MAZE-WM.json",
    "W16P2": "w16-maze-wm/PREREG-w16-maze-wm-T2.md",
    "W16C2R0": "w16-maze-wm/t2/C2r_val_seed0.json",
    "W16C2R1": "w16-maze-wm/t2/C2r_val_seed1.json",
    "W16C2R2": "w16-maze-wm/t2/C2r_val_seed2.json",
    "XS": "w15-crossover-seeds/CROSSOVER-3SEED.md",
    "XJ": "w15-crossover-seeds/CROSSOVER-3SEED.json",
    "XF": "w15-crossover-seeds/FINDINGS.md",
    "SD": "w15-single-design-seeds/T7-3SEEDS.md",
    "SDF": "w15-single-design-seeds/FINDINGS.md",
    "SD1": "w15-single-design-seeds/deletion/test_seed1_K32/deletion.json",
    "SD2": "w15-single-design-seeds/deletion/test_seed2_K32/deletion.json",
    "TF": "w15-table-fixes/FINDINGS.md",
    "TFX": "w15-table-fixes/TABLE-FIXES.md",
    "CT": "w15-cmaze-t-harness/FINDINGS.md",
    "CTJ": "w15-cmaze-t-harness/HARNESS-READY.json",
    "CO": "w15-cmaze-o-harness/FINDINGS.md",
    "D2B": "w15-maze-deletion-2b/TABLE.md",
    "D2F": "w15-maze-deletion-2b/FINDINGS.md",
    "RV": "w14-read-verb/FINAL-REPORT.md",
    "DRAFT": "/Users/batmac/clu/outputs/w17-paper/DRAFT-v2.tex",
    # pass-1 aliases re-used here (identical paths to build_tables.SRC)
    "KF": "w12b-nr6-and-kstar/FINDINGS.md",
    "W13T": "w13-seeds-and-tables/TABLES.md",
}
R = "ROW"

SPEC = [
    # ================= w16 PRIMARY: CMAZE-T Config A1, TEST (scored runs 1-3) =================
    ("a1_hdr", "W16T", 9, "Column `exact` (exact transition judge). n_seeds = 3. Config {'latent': 128, 'K': 2, 'n_atoms': 6, 'ridge': 1.0}, iface `none`, matched-rival steps 8000."),
    ("a1_n", "W16T", 12, "n per seed [761, 742, 686]"),
    ("a1n_clu", "W16T", 16, R), ("a1n_tf", "W16T", 25, R), ("a1n_mlp", "W16T", 23, R), ("a1n_lstm", "W16T", 24, R),
    ("a1n_rem", "W16T", 17, R), ("a1n_shuf", "W16T", 18, R), ("a1n_acr", "W16T", 19, R), ("a1n_ra", "W16T", 20, R),
    ("a1n_bs", "W16T", 22, R), ("a1n_bc", "W16T", 21, R),
    ("a1n_dmlp", "W16T", 26, R), ("a1n_dlstm", "W16T", 27, R), ("a1n_dtf", "W16T", 28, R),
    ("a1n_words", "W16T", 29, R), ("a1n_raw", "W16T", 30, R), ("a1n_frag", "W16T", 31, R), ("a1n_within", "W16T", 32, R),
    ("a1n_pbs", "W16T", 37, R), ("a1n_pbf", "W16T", 38, R), ("a1n_pbc", "W16T", 36, R),
    ("a1p_clu", "W16T", 44, R), ("a1p_tf", "W16T", 53, R), ("a1p_acr", "W16T", 47, R), ("a1p_bs", "W16T", 50, R), ("a1p_within", "W16T", 60, R),
    ("a1v_clu", "W16T", 72, R), ("a1v_tf", "W16T", 81, R), ("a1v_bs", "W16T", 78, R), ("a1v_within", "W16T", 88, R),
    ("a1s_clu", "W16T", 100, R), ("a1s_tf", "W16T", 109, R), ("a1s_words", "W16T", 113, R), ("a1s_within", "W16T", 116, R),
    ("a1_bars1", "W16F", 95, "SB1 PASS (margin 0.321, ahead every seed) · SB2 PASS (every control ≤ CLU − 0.15, bar"),
    ("a1_bars2", "W16F", 96, "0.575; also on each seed) · SB3 PASS (learned blind max 0.342 vs 0.675) · SB4 PASS."),
    ("a1_audit", "W16F", 98, "Replay 0 disagreements, 0 impossible rows, disjointness"),
    ("a1_seen_lbl", "W16F", 101, "`seen` 0.968 — sanity/retrieval (WORDS lookup ≈ 1.0 there)"),
    ("a1_pair_lbl1", "W16F", 102, "the control gate passes by **0.0001** (across-wells 0.2596 vs"),
    ("a1_pair_lbl2", "W16F", 103, "bar 0.2597, knife-edge) **but the learned BLIND-SEQUENTIAL arm reads 0.557 > CLU 0.410**"),
    ("a1_pair_lbl3", "W16F", 104, "**NOTHING is claimed on `pair`**"),
    ("a1_val_lbl", "W16F", 105, "`value` 0.465 **< best rival 0.511 and < learned blind-seq 0.485 — nothing"),
    ("t_link_pair", "W16F", 244, "**interface** (request design: a held-out pair cell is untrained; one-hot words carry no equality)"),
    ("t_link_val", "W16F", 245, "**content** (a never-trained colour has no trained cells)"),
    # rollout
    ("a1r_n", "W16T", 124, "[45, 45, 45] rollouts/seed"),
    ("a1r_clu", "W16T", 128, R), ("a1r_rem", "W16T", 129, R), ("a1r_mlp", "W16T", 133, R), ("a1r_lstm", "W16T", 134, R),
    ("a1r_tf", "W16T", 135, R), ("a1r_dmlp", "W16T", 136, R), ("a1r_dlstm", "W16T", 137, R), ("a1r_dtf", "W16T", 138, R),
    ("a1d_range", "W16F", 110, "store-scoped exactness 7.7e-16 … 1.2e-15 (S_k rel)"),
    ("c5d_range", "W16F", 176, "store-scoped exactness 8.1e-16 … 1.5e-15, atoms Δ 0.0"),
    ("t_roll_step_f", "W16F", 107, "(per-step exactness CLU 0.925 vs transformer"),
    ("t_roll_all", "W16F", 107, "**Rollout validity 0.000 for every learned arm**"),
    ("t_roll_arith", "W16F", 108, "0.818; 0.93^43 ≈ 0.04 — arithmetic, as w15 said"),
    ("t_link_roll", "W16F", 246, "**read** (per-step error compounds over 43 steps)"),
    ("t_link_roll_j", "W16J", 375, "\"T1 rollout\": \"read\""),
    # deletion A1
    ("a1d_hdr", "W16T", 140, "(deleted colour 1; store-scoped exactness; composition on surviving colours, nway)"),
    ("a1d_0v", "W16T", 144, R), ("a1d_0c", "W16T", 145, R), ("a1d_1v", "W16T", 146, R),
    ("a1d_1c", "W16T", 147, R), ("a1d_2v", "W16T", 148, R), ("a1d_2c", "W16T", 149, R),
    ("a1d_light", "W16F", 112, "**light and data-free** (A, Mx, S_k, t_k, n_k downdated, same closed"),
    ("a1d_steps", "W16F", 114, "and 8,000 gradient steps. Frozen codec and state encoder declared OUTSIDE the deletable object."),
    # ledger A1
    ("t_conv", "W16T", 151, "deployed bytes = 4 × (atom + address-book floats), float32; float64 sufficient statistic beside, never netted; rivals matched on read floats"),
    ("a1l_0", "W16T", 155, R), ("a1l_1", "W16T", 156, R), ("a1l_2", "W16T", 157, R),
    ("w16_2pct", "W16F", 238, "`rivals` block of each CMAZE-O JSON (all `matched: true`, within 2 %)"),
    ("w16_led_a1", "W16F", 233, R), ("w16_led_c5", "W16F", 234, R), ("w16_led_o", "W16F", 235, R),
    # ================= w16 PRIMARY: CMAZE-T Config C-512, TEST (scored runs 7-9) =================
    ("c5_hdr", "W16T", 160, "Column `exact` (exact transition judge). n_seeds = 3. Config {'latent': 512, 'K': 2, 'n_atoms': 8, 'ridge': 1.0}, iface `none`, matched-rival steps 80000."),
    ("c5n_clu", "W16T", 167, R), ("c5n_rem", "W16T", 168, R), ("c5n_shuf", "W16T", 169, R), ("c5n_acr", "W16T", 170, R),
    ("c5n_ra", "W16T", 171, R), ("c5n_bc", "W16T", 172, R), ("c5n_bs", "W16T", 173, R), ("c5n_mlp", "W16T", 174, R),
    ("c5n_lstm", "W16T", 175, R), ("c5n_tf", "W16T", 176, R), ("c5n_dmlp", "W16T", 177, R), ("c5n_dlstm", "W16T", 178, R),
    ("c5n_dtf", "W16T", 179, R), ("c5n_words", "W16T", 180, R), ("c5n_within", "W16T", 183, R),
    ("c5p_clu", "W16T", 195, R), ("c5p_bs", "W16T", 201, R), ("c5p_tf", "W16T", 204, R),
    ("c5v_clu", "W16T", 223, R), ("c5v_bs", "W16T", 229, R), ("c5v_mlp", "W16T", 230, R),
    ("c5s_clu", "W16T", 251, R), ("c5s_tf", "W16T", 260, R), ("c5s_words", "W16T", 264, R),
    ("c5r_clu", "W16T", 279, R), ("c5r_tf", "W16T", 286, R), ("c5r_dtf", "W16T", 289, R),
    ("c5d_0v", "W16T", 295, R), ("c5d_0c", "W16T", 296, R), ("c5d_1v", "W16T", 297, R),
    ("c5d_1c", "W16T", 298, R), ("c5d_2v", "W16T", 299, R), ("c5d_2c", "W16T", 300, R),
    ("c5l_0", "W16T", 306, R), ("c5l_1", "W16T", 307, R), ("c5l_2", "W16T", 308, R),
    ("c5_bars1", "W16F", 165, "SB1 PASS (margin 0.272 over 10×-trained rivals, ahead"),
    ("c5_bars2", "W16F", 166, "on every seed) · SB2 PASS (max control 0.448 ≤ 0.575) · SB3 PASS (0.414 ≤ 0.675) · SB4 PASS."),
    ("c5_cav1", "W16F", 169, "random_atoms reads 0.464 against the CLU's 0.567 (0.103 < 0.15)"),
    ("c5_cav2", "W16F", 172, "the headline store row; C-512 is its replication, with this caveat attached."),
    ("c5_cavj", "W16J", 368, "per seed: seed 1 random_atoms 0.464 vs CLU 0.567 (margin 0.103)"),
    ("c5_other", "W16F", 173, "`pair` 0.485 < learned blind-seq 0.637 → nothing; `value` 0.496 ≈ best rival"),
    ("c5_other2", "W16F", 174, "0.494 and < blind-seq 0.501 → nothing"),
    ("c5_replay", "W16F", 175, "Replay 0 disagreements on all seeds."),
    ("c5_steps", "W16F", 178, "**1,002–1,646 s and 80,000 gradient steps**"),
    # ================= w16 PRIMARY: CMAZE-T VAL training-budget diagnostic (A1, 80k steps) =================
    ("dg_hdr", "W16T", 1281, "CMAZE-T, A1 rows, rivals at 80,000 steps (10× recipe). Source `t1/A1_rivaldiag_val_seed*.json`. CLU A1 VAL nway 0.802 / 0.569 / 0.803."),
    ("dg_mlp", "W16T", 1285, R), ("dg_lstm", "W16T", 1286, R), ("dg_tf", "W16T", 1287, R),
    ("dg_mlp10", "W16T", 1288, R), ("dg_lstm10", "W16T", 1289, R), ("dg_tf10", "W16T", 1290, R),
    ("dg_close", "W16F", 120, "(closest: seed 1, lstm 0.437 vs"),
    # Config B, the plateau, AMENDMENT 1, selection
    ("b_val", "W16F", 125, "0.9258, pair 0.8443 — but the learned BLIND-SEQUENTIAL arm reads **0.8782 > bar 0.8758**."),
    ("b_link", "W16F", 126, "Named link: **INTERFACE** — the bits are the interaction."),
    ("b_lift", "W16F", 127, "CLU 0.7247 → 0.9258 (**+0.20**); best matched rival 0.3891 → 0.7902 (**+0.40**)."),
    ("plateau", "W16F", 137, "VAL n-way = 0.7247 with the SAME per-seed triple for ~15 configs spanning latent 64–512,"),
    ("plateau2", "W16F", 140, "That is why seed 1 caps at 0.569 everywhere"),
    ("plateau3", "W16F", 31, "n-way plateaus at 0.7247 across ~15 store configs; seed 1 at 0.569 in every"),
    ("am1_a", "W16F", 59, "run it **was** the identity on seeds 0 and 1: control == CLU to every digit (0.8019,"),
    ("am1_b", "W16F", 60, "0.5689; `t1/A_val_seed*.json`). The 3-seed mean (0.489) cleared the gate only because seed"),
    ("am1_c", "W16F", 65, "0.2504 on VAL n-way (was 0.489). ⭐ w15's own CMAZE-T numbers at K = 4/8 carry the same"),
    ("sel_1", "W16F", 74, "25/53 store-gate eligible; best VAL n-way per K = 0.7247 (K 2),"),
    ("sel_2", "W16F", 75, "0.7318 (K 4), 0.7247 (K 8), 0.7247 (K 16) → **K\\* = 2 → Config A1 = latent 128, K 2,"),
    # ================= w16 PRIMARY: CMAZE-O Config C-field, TEST (scored runs 4-6) =================
    ("o_hdr", "W16T", 755, "Column `valid` (exact objects AND legal). n_seeds = 3. Config {'K': 8, 'n_atoms': 64, 'ridge': 1.0, 'pair_block': True, 'space': 'field', 'dim': 2475, 'partition': 'ward_balanced'}, rival steps 3000."),
    ("o_n", "W16T", 758, "n per seed [400, 400, 400]"),
    ("on_ceil", "W16T", 762, R), ("on_look", "W16T", 763, R), ("on_clu", "W16T", 764, R), ("on_rem", "W16T", 765, R),
    ("on_shuf", "W16T", 766, R), ("on_chan", "W16T", 767, R), ("on_acr", "W16T", 768, R), ("on_ra", "W16T", 769, R),
    ("on_mlp", "W16T", 770, R), ("on_cmlp", "W16T", 771, R), ("on_lstm", "W16T", 772, R), ("on_tf", "W16T", 773, R),
    ("on_ae", "W16T", 774, R), ("on_b1", "W16T", 775, R), ("on_b2", "W16T", 776, R), ("on_b3", "W16T", 777, R),
    ("on_b4", "W16T", 778, R), ("on_rand", "W16T", 779, R),
    ("o2_n", "W16T", 781, "n per seed [396, 396, 396]"),
    ("o2_clu", "W16T", 787, R), ("o2_shuf", "W16T", 789, R), ("o2_ra", "W16T", 792, R), ("o2_acr", "W16T", 791, R),
    ("o2_mlp", "W16T", 793, R), ("o2_look", "W16T", 786, R), ("o2_bnl", "W16T", 801, R),
    ("os_n", "W16T", 804, "n per seed [600, 600, 600]"),
    ("os_clu", "W16T", 810, R), ("os_look", "W16T", 809, R), ("os_mlp", "W16T", 816, R), ("os_shuf", "W16T", 812, R),
    ("ov_n", "W16T", 827, "n per seed [400, 400, 400]"),
    ("ov_clu", "W16T", 833, R), ("ov_ceil", "W16T", 831, R), ("ov_bnl", "W16T", 847, R),
    ("ovs_clu", "W16T", 856, R),
    ("ov_lbl", "W16F", 272, "comparison: `value_holdout` is 0.000 for every arm incl. the field ceiling-bypassing CLU —"),
    ("ov_lbl2", "W16F", 273, "a judge-integrity row, as in w15."),
    ("o_field1", "W16F", 195, "the parameterless `decode_state`; no AE in the read path; every rival regresses the"),
    ("o_bars", "W16F", 202, "By the pre-registered rule: SB0–SB4 PASS."),
    ("o_ident1", "W16P2", 21, "⚠ The atom-ROW `landscape_shuffled` is a proved identity for a re-solved"),
    ("o_ident2", "W16P2", 22, "it is printed in `full` runs and"),
    ("o_ident3", "W16P2", 23, "never used as evidence either way"),
    ("o_gate_j", "W16J", 369, "\"T2 C-field TEST unseen_nway\": \"PASS: all four controls 0.000 on every seed; the STORE is load-bearing, but the composition-over-rivals claim is withdrawn\""),
    ("o_verdict_j", "W16J", 22, "WIN by the pre-registered rule; rival margin WITHDRAWN by the VAL training-budget diagnostic (matched MLP at 10x steps 0.99 vs CLU 0.70); store gate holds (controls 0.000); INTERFACE finding"),
    # deletion C-field
    ("od_hdr", "W16T", 873, "(colour 2; store-scoped; unseen_nway)"),
    ("od_0vo", "W16T", 877, R), ("od_0vd", "W16T", 878, R), ("od_0co", "W16T", 879, R), ("od_0cd", "W16T", 880, R),
    ("od_1vo", "W16T", 881, R), ("od_1vd", "W16T", 882, R), ("od_1co", "W16T", 883, R), ("od_1cd", "W16T", 884, R),
    ("od_2vo", "W16T", 885, R), ("od_2vd", "W16T", 886, R), ("od_2co", "W16T", 887, R), ("od_2cd", "W16T", 888, R),
    ("od_find", "W16F", 205, "relative ΔS = 0.0 on every seed; after-deletion == refit-without in every"),
    ("od_find2", "W16F", 208, "CLU adaptation 1.8–3.3 s (light, data-free) vs MLP retrain-without 27.7–29.3 s."),
    # ledger C-field
    ("ol_0", "W16T", 890, "seed 0 read floats 1232064 (atoms 158400 + head 1073664), rank/width 0.1742013780144065, lookup items 497, rivals {'mlp': 1231290, 'cont_mlp': 1233449, 'lstm': 1231661, 'transformer': 1217195}"),
    ("ol_1", "W16T", 890, "rank/width 0.18226458920555383"),
    ("ol_2", "W16T", 890, "rank/width 0.17581428124021298"),
    # the withdrawal diagnostic (C-field vs 30k-step rivals, VAL)
    ("ow_hdr", "W16T", 1292, "CMAZE-O, C-field rows, matched rivals at 30,000 steps (10× recipe). Source `t2/C_rivaldiag_val_seed*.json`. CLU C-field VAL unseen_nway 0.640 / 0.730 / 0.730."),
    ("ow_mlp", "W16T", 1296, R), ("ow_cmlp", "W16T", 1297, R), ("ow_lstm", "W16T", 1298, R), ("ow_tf", "W16T", 1299, R),
    ("ow_find", "W16F", 212, "unseen n-way: **mlp 0.988 / 0.990 / 0.990**, transformer 0.748 / 0.643 / 0.628, lstm"),
    ("ow_survive", "W16F", 215, "What survives: (i) the store gate (controls 0.000 —"),
    ("ow_survive2", "W16F", 216, "a STORE row relative to its controls), (ii) the read-space effect for the CLU (field"),
    ("ow_survive3", "W16F", 217, "0.69 vs latent ≤ 0.14 on the same rows), (iii) exact deletion. **Named link: INTERFACE** —"),
    ("o_c_val_f", "W16F", 17, "WIN by the rule (0.693 vs 0.167)"),
    ("ow_need2", "W16F", 215, "(it would need ≥ 1.02)"),
    ("ow_need", "W16F", 214, "**No CMAZE-O config can pass VB1 against a converged"),
    # C2 (latent) VAL vs 30k-step rivals
    ("c2_hdr", "W16T", 1164, "Config {'K': 2, 'n_atoms': 32, 'ridge': 1.0, 'pair_block': True, 'space': 'latent', 'dim': 128, 'partition': 'ward_balanced'}, rival steps 30000."),
    ("c2_clu", "W16T", 1173, R), ("c2_rem", "W16T", 1174, R), ("c2_shuf", "W16T", 1175, R), ("c2_chan", "W16T", 1176, R),
    ("c2_acr", "W16T", 1177, R), ("c2_ra", "W16T", 1178, R), ("c2_mlp", "W16T", 1179, R), ("c2_lstm", "W16T", 1181, R), ("c2_tf", "W16T", 1182, R),
    ("c2_find", "W16F", 220, "FAILS VAL** against them: CLU 0.088 / 0.150 / 0.173 vs rival 0.315 / 0.413 / 0.463"),
    ("c2_find0", "W16F", 219, "PREREG-T2 AMENDMENT 4 then made 30k-step rivals mandatory; **C2 (latent, rank/width 0.5)"),
    ("c2_j0", "W16C2R0", 281, "\"valid\": 0.0875,"),
    ("c2_j1", "W16C2R1", 281, "\"valid\": 0.15,"),
    ("c2_j2", "W16C2R2", 281, "\"valid\": 0.1725,"),
    ("c2_led", "W16T", 1277, "seed 0 read floats 138304 (atoms 4096 + head 134208), rank/width 0.484375"),
    # CMAZE-O Config A, failing links, budget, not-run
    ("oa_best", "W16F", 187, "best unseen n-way **0.0208**"),
    ("oa_link", "W16F", 188, "**Named link: CONTENT/READ**"),
    ("lk_pair", "W16J", 373, "\"T1 pair\": \"interface\""),
    ("lk_val", "W16J", 374, "\"T1 value\": \"content\""),
    ("lk_b", "W16J", 376, "\"T1 Config B\": \"interface\""),
    ("lk_oa", "W16J", 377, "\"T2 Config A\": \"content/read\""),
    ("lk_oc", "W16J", 378, "\"T2 C-field vs converged rivals\": \"interface\""),
    ("lk_c2", "W16J", 379, "\"T2 C2\": \"read\""),
    ("bud_1", "W16F", 258, R), ("bud_4", "W16F", 259, R), ("bud_7", "W16F", 260, R), ("bud_10", "W16F", 261, R), ("bud_13", "W16F", 262, R),
    ("w16_nr1", "W16F", 267, "A 10×-steps re-scoring of the C-field TEST"),
    ("w16_nr2", "W16F", 268, "Deletion rows for"),
    ("w16_nr3", "W16F", 269, "Config B / C2 / C2r (VAL only, not winners; `DELETION: DECLARED_NOT_RUN` in the JSONs)."),
    ("w16_nr4", "W16F", 270, "GPU/cluster; any seed beyond 3."),
    ("w16_seeds", "W16F", 5, "**n_seeds = 3 (0, 1, 2) on"),
    # ================= w15 crossover-seeds -> tab:kcurve =================
    ("xs_conv1", "XS", 11, "(1) atoms = sum_k modes_k.size x 4 B (float32, the deployed content);"),
    ("xs_conv2", "XS", 12, "(2) covariance + count = sum_k stat_floats_k x 8 B (float64 -- the precision actually retained;"),
    ("xs_conv3", "XS", 14, "(3) address book = TensorAddress.byte_ledger()['address_bytes'];"),
    ("xs_conv4", "XS", 15, "(4) coupling/hole table = 5,288 + 1,322 = 6,610 B."),
    ("xs_k5", "XS", 21, R), ("xs_k8", "XS", 22, R), ("xs_k10", "XS", 23, R), ("xs_k12", "XS", 24, R), ("xs_k16", "XS", 25, R),
    ("xs_k20", "XS", 26, R), ("xs_k24", "XS", 27, R), ("xs_k28", "XS", 28, R), ("xs_k32", "XS", 29, R),
    ("xs_rule", "XS", 31, "**K\\* = 32**, **K_crossover = None** -- no K in the swept range drops more than 0.1 below K=5 at n = 3"),
    ("xs_l20_0", "XS", 68, R), ("xs_l20_1", "XS", 69, R), ("xs_l20_2", "XS", 70, R),
    ("xs_l24_0", "XS", 71, R), ("xs_l24_1", "XS", 72, R), ("xs_l24_2", "XS", 73, R),
    ("xs_l28_0", "XS", 74, R), ("xs_l28_1", "XS", 75, R), ("xs_l28_2", "XS", 76, R),
    ("xs_l32_0", "XS", 77, R), ("xs_l32_1", "XS", 78, R), ("xs_l32_2", "XS", 79, R),
    ("xs_s3", "XS", 100, "**S3 VERDICT: CONFIRMED**"),
    ("xs_minima", "XS", 97, "per-seed minima: `{'0': 24, '1': 24, '2': 24}`"),
    ("xs_s4", "XS", 101, "S4 resolvability at K = 24: seed band sd 7,060 B; neighbour gaps K=20 12,203 B (clears the band), K=28 30,507 B (clears the band); minimum resolvable: **True**"),
    ("xs_c20", "XS", 112, R),
    ("xs_c20cell", "XS", 123, R),
    ("xs_gate20", "XS", 134, R),
    ("xs_ex20", "XS", 150, R), ("xs_ex32", "XS", 153, R),
    ("xf_nores", "XF", 144, "### 2b. ⛔ The frozen rule is NOT resolvable at n = 3, and that is said plainly"),
    ("xf_nores2", "XF", None, "**K\\* = 32 is what the rule returns; it is not a K that three seeds can"),
    ("xf_drop0", "XF", None, "**Every drop at n = 3 is ≤ 0: the largest drop anywhere in the sweep is 0.0000, at K = 5"),
    ("xf_w12wrong", "XF", None, "w12's \"largest drop anywhere is 0.0093 at K = 10\" was a seed-0 artefact."),
    ("xf_floor", "XF", None, "the floor over"),
    ("xf_floor2", "XF", None, "the whole sweep and all three seeds is 3.9675 × the reference, at K = 24 seed 1**"),
    ("xf_seeds34", "XF", None, "**Seeds 3 and 4: DECLARED NOT-RUN.**"),
    ("xj_m8", "XJ", None, "\"mean\": 0.9037422839506172,"),
    ("xj_d8", "XJ", None, "\"drop_from_K5\": -0.008487654320987525,"),
    # ================= w15 single-design-seeds -> tab:del, tab:rivals-del, tab:hole =================
    ("sd_t7a", "SD", 15, R), ("sd_t7b", "SD", 16, R),
    ("sd_gate0", "SD", 42, R), ("sd_gate1", "SD", 43, R), ("sd_gate2", "SD", 44, R),
    ("sd_led0", "SD", 71, R), ("sd_led1", "SD", 72, R), ("sd_led2", "SD", 73, R),
    ("sd_v1", "SD", 86, R), ("sd_v2", "SD", 87, R), ("sd_cl1", "SD", 92, R), ("sd_cl2", "SD", 93, R),
    ("sd_i1", "SD", 95, R), ("sd_i2", "SD", 96, R), ("sd_c1", "SD", 89, R), ("sd_c2", "SD", 90, R),
    ("sd_rv1", "SD", 105, R), ("sd_rv2", "SD", 107, R), ("sd_rcl1", "SD", 117, R), ("sd_rcl2", "SD", 119, R),
    ("sd_ri1", "SD", None, "| `item` | 1 | refit-without (the store reference) | 17.624 | 129,761 | 1,065,216 | yes |"),
    ("sd_ri2", "SD", None, "| `item` | 2 | refit-without (the store reference) | 18.649 | 129,761 | 1,065,216 | yes |"),
    ("sd_rmlp0", "SD", 135, R), ("sd_rlstm0", "SD", 136, R), ("sd_rtf0", "SD", 137, R), ("sd_rcm0", "SD", 138, R),
    ("sd_rmlp1", "SD", 139, R), ("sd_rlstm1", "SD", 140, R), ("sd_rtf1", "SD", 141, R), ("sd_rcm1", "SD", 142, R),
    ("sd_rmlp2", "SD", 143, R), ("sd_rlstm2", "SD", 144, R), ("sd_rtf2", "SD", 145, R), ("sd_rcm2", "SD", 146, R),
    ("sd1_cv", "SD1", 282, "\"principal_angle_cos_min\": 0.04069122680502917,"),
    ("sd1_ccl", "SD1", 3095, "\"principal_angle_cos_min\": 0.20550653338433075,"),
    ("sd1_ci", "SD1", 4494, "\"principal_angle_cos_min\": 0.06958072632551195,"),
    ("sd2_cv", "SD2", 282, "\"principal_angle_cos_min\": 0.28057461977018805,"),
    ("sd2_ccl", "SD2", 3088, "\"principal_angle_cos_min\": 0.618318259716018,"),
    ("sd2_ci", "SD2", 4487, "\"principal_angle_cos_min\": 0.18117547035211387,"),
    ("sdf_bytes1", "SDF", 245, "because ours counts the atoms"),
    ("sdf_bytes2", "SDF", 246, "re-derived **plus** the covariance/count downdated **plus** the address book carried,"),
    ("sdf_bytes3", "SDF", 247, "while the reference rewrites only the atoms and the statistic."),
    ("sdf_hole1", "SDF", 273, "in a **1,322 B** coupling table, cost **2 bytes"),
    ("sdf_hole2", "SDF", 274, "rewritten, 0.000 s, 0 data rows read, no refit**, at all three seeds"),
    ("sdf_red0", "SDF", 290, R), ("sdf_red1", "SDF", 291, R), ("sdf_red2", "SDF", 292, R),
    ("sdf_redclaim", "SDF", None, "The deletion claim here is store-scoped exactness plus a data-free"),
    ("sdf_not1x", "SDF", 100, "⛔ No row here is reported as a 1× result."),
    ("sdf_scale", "SDF", 103, "`BASELINES-READY.json → per_seed.*.build_scale.n_fit_rows_used = 16,384`"),
    ("sdf_spent", "SDF", 21, "⭐ **I spent ZERO scored runs.**"),
    # ================= w15 maze-deletion-2b -> new tab:maze-del-2b =================
    ("d2_env", "D2B", 3, "**This table contains no CMAZE-1 or CMAZE-2 row.** Every row: environment **CMAZE-2b**. Seeds = 3."),
    ("d2_0", "D2B", 7, R), ("d2_1", "D2B", 8, R), ("d2_2", "D2B", 9, R),
    ("d2_a0", "D2B", 15, R), ("d2_a0c", "D2B", 16, R), ("d2_a1", "D2B", 17, R), ("d2_a1c", "D2B", 18, R),
    ("d2_a2", "D2B", 19, R), ("d2_a2c", "D2B", 20, R),
    ("d2_r0", "D2B", 26, R), ("d2_r1", "D2B", 27, R), ("d2_r2", "D2B", 28, R),
    ("d2_led1", "D2F", 73, "the read has 410688 floats = 1642752 bytes at"),
    ("d2_led2", "D2F", 74, "**float32 deployed** (atoms 7936 + address book 402752). The sufficient statistic is 35904 bytes"),
    ("d2_led3", "D2F", 75, "at **float64 retained**"),
    ("d2_proj", "D2F", 71, "number is the **float64 statistic residual** (7.2e-16 – 9.1e-16). Quote the two together."),
    ("d2_norefuse", "D2F", 126, "**Deletion does not induce refusal of the deleted word.**"),
    ("d2_norefuse2", "D2F", 128, "claim \"deleted ⇒ refused\". The claim is store-scoped exactness, nothing more."),
    ("d2_speed", "D2F", 145, "own store and head from the survivors (≈ 0.8–0.9 s vs ≈ 1.1–1.2 s). The differentiator is not"),
    ("d2_carry_s", "D2F", None, "| CMAZE-2b | 0 | basis-change carry W ← WR (w9 as pre-registered) | 0.009 |"),
    ("d2_nr1", "D2F", 158, "**Test-leg adaptation composition: DECLARED NOT-RUN.**"),
    ("d2_nr2", "D2F", 161, "**Rival retrain-without for the MLP / LSTM: DECLARED NOT-RUN.**"),
    ("d2_extent", "D2F", None, "every atom and ≈ 99.98 % of the address book"),
    # ================= w15 harnesses -> P12 =================
    ("ct_rep", "CT", 103, R),
    ("ct_rep0", "CT", 100, R), ("ct_rep1", "CT", 101, R), ("ct_rep2", "CT", 102, R),
    ("ct_imp", "CT", 119, "**164,767 rows checked, 0 with no accepting emission.**"),
    ("ct_w1", "CTJ", 280, "\"warning\": \"composition_blind_PLANNER_scores_near_the_clu_on_unseen_nway\","),
    ("ct_w2", "CTJ", 289, "\"warning\": \"charter_C_control_did_not_collapse\","),
    ("ct_w3", "CTJ", 298, "\"warning\": \"cross_maze_lookup_floor_scores_near_the_clu\","),
    ("ct_vn", "CT", 233, R), ("ct_v2", "CT", 208, R), ("ct_vs", "CT", 182, R), ("ct_vv", "CT", 259, R),
    ("ct_g2", "CT", 403, R),
    ("co_rep", "CO", 103, "**105,408 states replayed, 0 disagreements**, 35,136 per seed; layout-bank rebuild mismatches **0**"),
    ("co_imp", "CO", 104, "**30,588 requests checked, 0 impossible** (10,196 per seed)"),
    ("co_ctrl", "CO", 25, "**the charter-C controls do not collapse.** At the span fraction Config-A deploys"),
    ("co_iface", "CO", 27, "**INTERFACE claim, not a STORE claim**"),
    ("co_iface2", "CO", 28, "and the interface here is 201,312 of 202,848 read floats (99.24 %)."),
    # ================= w15 table-fixes -> MF2, MF14(i), against DRAFT-v2 =================
    ("tf_mf2", "TFX", 84, "**23.5121×**. The referee's figure is a rounding slip in the 4th decimal."),
    ("tf_mf2b", "TFX", 83, "statistic's ratio as **23.5117×**."),
    ("dr_1317", "DRAFT", 1317, "the statistic alone is 23.5117 times"),
    ("dr_2262", "DRAFT", 2262, "its statistic alone is 23.5117 times"),
    ("tf_mf14", "TF", None, "### 2a. The BLIND columns that moved into the maze main-text table (MF14 i)"),
    ("tf_bs", "TF", None, "| `unseen_pair` | `BLIND-SEQUENTIAL(key order from A)` | 0.9077 / 0.8905 / 0.9128 | **0.9037** |"),
    ("dr_cmaze2_cols", "DRAFT", 1573, "row & seeds & store & MLP & LSTM & transformer & lookup (floor) & \\texttt{lscape\\_shuffled} & \\texttt{random\\_atoms} & \\texttt{lscape\\_removed} \\\\"),
    ("dr_cmaze2_prose", "DRAFT", 1591, "BLIND-SEQUENTIAL 0.9037 (0.9077 / 0.8905 / 0.9128) on unseen pairs and 0.9001 on $n$-way;"),
    ("tf_open1", "TF", None, "**`fig1_cmaze2_main_scoreboard.png` no longer matches its table.**"),
    ("tf_n14", "TF", None, "**N1–N4 remain open**"),
    # ================= draft cross-checks and the convention audit =================
    ("dr_t7a", "DRAFT", None, "\\textbf{single design, $\\Kstar=32$} & 3 & \\textbf{31.6000 / 32.2333 / 33.2167} & \\textbf{66.9592 / 65.9796 / 68.5918} & --- \\\\"),
    ("dr_del_bytes", "DRAFT", 555, "Every row rewrites 2,081,024 bytes of the store and address book, reads 0 data rows"),
    ("dr_kc_drop", "DRAFT", None, "crossover threshold of 0.10 is never reached; the largest drop anywhere is $+0.012731$ at $K=10$."),
    ("dr_kc_k8", "DRAFT", None, "8 & 1 & 0.883681 & +0.011574 & 2,133,280 & 295,424 & 5.0815 & 2,959,602 & 5.6450 \\\\"),
    ("dr_1x_cap", "DRAFT", 1164, "\\caption{\\textbf{$\\Kstar$ replication against a $1\\times$-total-bytes MLP (stress comparison),"),
    ("dr_hole_cap", "DRAFT", 624, "\\caption{\\textbf{Refusal DISPLAY rows. A display is never a result.} Our side is a cost line"),
    ("dr_hole_row", "DRAFT", None, "ours: 2 holes, $\\Kstar$ & 1 & 2 & 0 & --- & --- & --- \\\\"),
    ("dr_rd_cap", "DRAFT", 1406, "seed 0.} Data rows read and bytes rewritten are seed-invariant."),
    ("dr_rd_mlp", "DRAFT", None, "MLP & 1 / 3 & 8.86 & 15,032 & 522,056 & +0.03661 \\\\"),
    ("dr_verb_row", "DRAFT", 825, "\\multicolumn{2}{l}{gain, lookup floor (matched bytes)} & +0.8299 & 3 \\\\"),
    ("dr_cmazeo_cap", "DRAFT", None, "Store against MLP / LSTM / transformer / continuous MLP at matched parameters on"),
    ("bc_tri", "KF", 133, "symmetric-TRIANGLE float64"),
    ("bc_hole", "W13T", 431, "coupling table + one byte per hole"),
    ("bc_md_at", "W13T", 162, "store_atom_floats x 4 (float32)"),
    ("bc_md_ab", "W13T", 164, "head_floats x 4 (float32)"),
    ("rv_match1", "RV", 262, "⚠ **DECLARED — the byte match is not binding at this configuration.** The store's 410,688"),
    ("rv_match2", "RV", 264, "so `lookup_matched` (369,152 / 366,080 / 371,456 floats) **is** `lookup_full`: the floor is"),
    ("rv_match3", "RV", 266, "⛔ **The floor must be read as \"keep everything\", not as \"a compressed rival\".**"),
]


def main(out):
    lines_cache, rows, errs = {}, [], []
    for cid, alias, line, val in SPEC:
        path = Path(SRC2[alias]) if SRC2[alias].startswith("/") else EV / SRC2[alias]
        L = lines_cache.setdefault(alias, path.read_text(encoding="utf-8").splitlines())
        if line is None:
            hits = [i + 1 for i, t in enumerate(L) if val in t]
            if len(hits) != 1:
                errs.append(f"{cid}: {len(hits)} hits for {val!r} in {alias}: {hits[:8]}"); continue
            line = hits[0]
        text = L[line - 1]
        if val == R:
            val = text.strip().strip("|").strip()
        if val not in text:
            errs.append(f"{cid}: {val!r} not on {alias}:{line} -> {text[:150]!r}"); continue
        rows.append((cid, alias, line, val))
    if errs:
        print("\n".join(errs)); sys.exit(1)
    ids = [r[0] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate id"
    with open(out, "w", encoding="utf-8") as f:
        f.write('"""Pass-2 cell register (static; generated once from the spec, then re-verified by every build).\n\n'
                'Each cell is (source alias, 1-indexed line, value); build_tables.py fails unless the value is\n'
                'verbatim on that line. Row cells copy a whole markdown table row (outer pipes stripped)."""\n\n')
        f.write("SRC2 = {\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in SRC2.items()) + "}\n\n")
        f.write("CELLS2 = {\n" + "".join(f"    {c!r}: ({a!r}, {l}, {v!r}),\n" for c, a, l, v in rows) + "}\n")
    print(f"wrote {len(rows)} cells to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
