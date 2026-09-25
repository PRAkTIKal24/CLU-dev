"""w22 seeds-and-tables PASS 5: the w15 fold. Every number the drafts print under a w15-{bar-d-score,
coverage-control,lace-poe-rows} SRC scope, mapped to the artefact that holds it.

MAP: line anchor (a substring unique to one draft line) -> one entry per number token on that line, in the order
scan_draft.py finds them. An entry is (artefact, pointer) where artefact is a path under the evidence root and
pointer is an RFC-6901 JSON pointer, or ("TXT", path, verbatim) for a plain-text artefact line holding the value,
or ("NOJSON", reason) where the only artefact is prose (the reason is printed as a finding).
EXTRA: numbers the token scanner cannot see (integer zeros in table cells, counts split across lines), each with
its own draft anchor and verbatim draft string.
Nothing here is computed. The build checks each JSON value at the printed precision."""

CC = "w15-coverage-control/COVERAGE-CONTROL.json"
IC = "w15-coverage-control/IDENTITY-CHECK.txt"
G = "w15-lace-poe-rows/GENTABLE-KSTAR.json"
BD = "w15-bar-d-score/BAR-D.json"
XG = "w15-bar-d-score/CROSS-ARM-GPU-test.json"
W8P = "w8-encoder-loop/cloud/manifests/seed_{s}/alone/probe.json"
DC = "/test_context/C_test_DERIVED_not_measured"
SV = "/summary_val"
PA = "/PART_A_generation_table"
T = PA + "/table/"
R34 = "/PART_B_rows_3_4"
LAD = "/PART_B_ladder/k={k}/arms/{arm}/per_seed/{s}/all"


def seeds(f, fmt, n=3):
    return [(f, fmt.format(s=s)) for s in range(n)]


def gen_row(arm, extra_bytes=()):
    return seeds(G, T + arm + "/B_aggregate/per_seed/{s}") + [(G, T + arm + "/B_aggregate/mean")] + list(extra_bytes)


MAP = {
    # ---------------- coverage-control
    r"\emph{60.150 / 62.783 / 61.908, derived}": seeds(CC, DC + "/value/per_seed/{s}"),
    "On TEST, $n=6$ at $n=3$'s coverage is therefore 61.614": [(CC, DC + "/value/mean"), (CC, "/test_context/B_test_n6/mean")],
    "the whole $n=3\\to6$ rise of +4.989": [(CC, DC + "/coverage_B_minus_C_test/mean")] + seeds(CC, DC + "/coverage_B_minus_C_test/per_seed/{s}"),
    "acceptance 1.000000 under either rung's bitmap": [("TXT", IC, "accept fraction 1.000000")],
    "in 3,600 of 3,600 published TEST rows": [("TXT", IC, "rows checked: 3600"), ("TXT", IC, "rows checked: 3600"), ("TXT", IC, "True 8640")],
    "equals $n=3$ to every printed decimal on every seed, 22.767": [(CC, SV + "/store/C/mean")],
    "(21.767 / 23.050 / 23.483), against $n=6$'s 23.478": seeds(CC, SV + "/store/C/per_seed/{s}") + [(CC, SV + "/store/B/mean")] + seeds(CC, SV + "/store/B/per_seed/{s}"),
    "share of the rise is 1.00 on every seed and the mask channel is 0.000": [(CC, SV + "/coverage_share_s/mean"), (CC, SV + "/mask_C_minus_A/mean")],
    "+1.200 / +0.567 / +0.367, is below the pre-declared 1.0 margin": seeds(CC, SV + "/rise_B_minus_A/per_seed/{s}") + [(CC, "/estimand/margin")],
    # ---------------- lace-poe-rows: main text
    "score at most 0.267 and 0.020 distinct valid novel tuples": [(G, R34 + "/arms/LACE/per_seed/2/row3_pair"), (G, R34 + "/arms/PoE/per_seed/0/row4_nway")],
    "store's read (every seed, margin 0.020)": [(G, PA + "/bars/S2_vs_clu/margin"), (G, PA + "/bars/S2_vs_clu/delta_per_seed/0"), (G, PA + "/bars/S2_vs_clu/mean_delta")],
    "S3 \\textsc{lose}) at 4.09": [(G, PA + "/bars/S3_vs_knn_proto/mean_delta"), (G, PA + "/kstar_val_gates/0/ledger/times_reference_pin"), (G, PA + "/kstar_val_gates/2/ledger/times_reference_pin")],
    "($+0.4468$) and PoE ($+0.5532$) on the mean (S4)": [(G, PA + "/bars/S4_vs_lace/mean_delta"), (G, PA + "/bars/S4_vs_ebm_compose/mean_delta")],
    # ---------------- lace-poe-rows: tab:gentable-kstar
    r"\textbf{$\Kstar$ single design} & 3 &": gen_row("kstar") + [(G, T + "kstar/B_aggregate/std")] + seeds(G, PA + "/kstar_val_gates/{s}/ledger/honest_total_bytes") + [(G, PA + "/kstar_val_gates/0/ledger/times_reference_pin"), (G, PA + "/kstar_val_gates/2/ledger/times_reference_pin")],
    r"\quad \texttt{landscape\_removed} & 3 & 0.0556": gen_row("kstar~1landscape_removed"),
    r"\quad \texttt{landscape\_shuffled} & 3 &": gen_row("kstar~1landscape_shuffled"),
    r"\quad \texttt{landscape\_shuffled\_across\_wells} & 3 &": gen_row("kstar~1landscape_shuffled_across_wells"),
    r"\quad \texttt{random\_atoms} & 3 & 0.1458": gen_row("kstar~1random_atoms"),
    "earlier store's read & 3 & 0.5903": gen_row("clu") + [(G, T + "clu/B_aggregate/std"), (G, T + "clu/byte_ledger/deployed_bytes")],
    r"\texttt{store\_removed} & 3 & 0.1111": gen_row("store_removed") + [(G, T + "store_removed/byte_ledger/budget_bytes")],
    "nearest-prototype ceiling & 3 & 0.8588": gen_row("knn_proto") + [(G, T + "knn_proto/B_aggregate/std"), (G, T + "knn_proto/byte_ledger/deployed_bytes")],
    r"LACE (\texttt{ode}) & 3 & 0.2940": gen_row("lace") + [(G, T + "lace/B_aggregate/std")],
    r"PoE (\texttt{ebm\_compose}) & 3 & 0.1759": gen_row("ebm_compose") + [(G, T + "ebm_compose/B_aggregate/std"), (G, T + "ebm_compose/byte_ledger/deployed_bytes"), (G, R34 + "/arms/PoE/budget_per_seed/0/x_store_reference")],
    "earlier store's read (S2) & 3 &": seeds(G, PA + "/bars/S2_vs_clu/delta_per_seed/{s}") + [(G, PA + "/bars/S2_vs_clu/mean_delta")],
    "nearest-prototype ceiling (S3) & 3 &": seeds(G, PA + "/bars/S3_vs_knn_proto/delta_per_seed/{s}") + [(G, PA + "/bars/S3_vs_knn_proto/mean_delta")],
    # ---------------- lace-poe-rows: tab:lace-poe caption and blocks
    "their aggregates, 0.3032 and 0.1968, sit at the": [(G, T + "lace/B_aggregate/mean"), (G, T + "ebm_compose/B_aggregate/mean")],
    "landscape controls (0.1088 to 0.1860)": [(G, T + "kstar~1landscape_shuffled_across_wells/B_aggregate/mean"), (G, T + "kstar~1random_atoms/B_aggregate/mean")],
    "(0.1204), and neither renders": [(G, T + "store_removed/B_aggregate/mean")],
    "LACE (own budget, not matched) & 3 & 0.117": seeds(G, R34 + "/arms/LACE/per_seed/{s}/row3_pair") + seeds(G, R34 + "/arms/LACE/per_seed/{s}/row4_nway") + seeds(G, R34 + "/verdict/LACE/mean_paired_delta_rival_minus_kstar_per_seed/{s}"),
    "PoE (own budget, not matched) & 3 & 0 / 0 / 0": [(G, R34 + "/arms/PoE/per_seed/0/row4_nway")] + seeds(G, R34 + "/verdict/PoE/mean_paired_delta_rival_minus_kstar_per_seed/{s}"),
    "each runs at its own declared budget (LACE 31.08": [(G, R34 + "/arms/LACE/budget_per_seed/0/x_store_reference_classifier_only"), (G, R34 + "/arms/PoE/budget_per_seed/0/x_store_reference"), (G, T + "clu/byte_ledger/budget_bytes"), ("NOJSON", "a count: corners_control_wins = n_corners = 109 at every seed, checked in the EXTRA cells")],
    "LACE (own budget, not matched) & 3 & 0.067": [(G, LAD.format(k=k, arm="LACE", s=s)) for k in range(2, 7) for s in range(3)],
    "PoE (own budget, not matched) & 3 & 0.017": [(G, LAD.format(k=k, arm="PoE", s=s)) for k in range(2, 7) for s in range(3)],
    "store on rows 3--4, nor anywhere on the ladder: on rows 3--4 the $\\Kstar$ store scores 47.50": [(G, R34 + "/arms/kstar_store/per_seed/0/all")],
    "47.40 / 49.12 novel valid tuples per corner against LACE": [(G, R34 + "/arms/kstar_store/per_seed/1/all"), (G, R34 + "/arms/kstar_store/per_seed/2/all")] + seeds(G, R34 + "/arms/LACE/per_seed/{s}/all") + [(G, R34 + "/arms/PoE/per_seed/0/all")],
    "0.000 / 0.000, and the store wins 109/109 corners": [(G, R34 + "/arms/PoE/per_seed/1/all"), (G, R34 + "/arms/PoE/per_seed/2/all"), ("NOJSON", "a count: see the EXTRA cells for corners_control_wins")],
    "(60 of 60 at $n=2$, 120 of 120 at $n\\ge3$)": [("NOJSON", "a count: see the EXTRA cells"), ("NOJSON", "a count: see the EXTRA cells")],
    "LACE 4,073,663 classifier parameters": [(G, R34 + "/arms/LACE/budget_per_seed/0/classifier_params")],
    "(31.08$\\times$ the 524,288 B reference; 63.10": [(G, R34 + "/arms/LACE/budget_per_seed/0/x_store_reference_classifier_only"), (G, T + "clu/byte_ledger/budget_bytes"), (G, R34 + "/arms/LACE/budget_per_seed/0/x_store_reference_with_whitening")],
    "experts with 3,556,665 parameters": [(G, R34 + "/arms/PoE/budget_per_seed/0/expert_params"), (G, R34 + "/arms/PoE/budget_per_seed/0/x_store_reference"), (G, R34 + "/arms/LACE/budget_per_seed/0/fit_rows")],
    # ---------------- bar-d-score
    "The unfreezing decomposition on the longer leg only (4,000 joint steps": [(BD, "/BUDGET_PINS_OF_THIS_ROW/joint_steps")],
    "32,768 training images, 1,800 decoder-refit steps, 4,000 warm steps": [(BD, "/BUDGET_PINS_OF_THIS_ROW/n_train"), (BD, "/BUDGET_PINS_OF_THIS_ROW/dec_refit_steps"), (BD, "/BUDGET_PINS_OF_THIS_ROW/warm_steps")],
    r"store: the \texttt{pca\_tensor} atom block (524,288 B, float32)": [("w15-bar-d-score/score/alone/test_seed0/READ-TABLE.json", "/arms/pca_tensor/store_bytes")],
    r"$A'$ (unfreeze: \texttt{loop\_nosw} $-$ \texttt{alone}) & 3 &": [(XG, f"/per_seed/{s}/{r}/encoder_comparison/BAR_Aprime_loop_nosw_minus_alone/mean_paired_delta") for r in ("r3_comp_2way", "r4_comp_nway") for s in range(3)],
    r"$A''$ (swap loss: \texttt{loop\_swap} $-$ \texttt{loop\_nosw}) & 3 &": [(XG, f"/per_seed/{s}/{r}/encoder_comparison/BAR_Adprime_swap_loss_alone/mean_paired_delta") for r in ("r3_comp_2way", "r4_comp_nway") for s in range(3)],
    "ordering (pre-declared rule) & 3 &": [(BD, "/ORDERING/GPU_summary/Aprime_negative"), (BD, "/ORDERING/GPU_summary/Adprime_positive")],
    "negative in only 2 of 6 cells, while $A''$ is positive in 6 of 6": [(BD, "/ORDERING/GPU_summary/Aprime_negative"), (BD, "/ORDERING/GPU_summary/Adprime_positive")],
    "6). The swap-loss half survives and strengthens (positive in 4 of 6": [(BD, "/ORDERING/CPU_summary/Adprime_positive")],
    r"\texttt{alone} control trained well (probe margin over random initialisation $-0.0505$": [(W8P.format(s=0), "/MARGIN_OVER_RANDOM_INIT/aggregate")],
    "$+0.0242$ / $-0.0862$; row-4 score 38.5 / 64.4 / 39.3": [(W8P.format(s=1), "/MARGIN_OVER_RANDOM_INIT/aggregate"), (W8P.format(s=2), "/MARGIN_OVER_RANDOM_INIT/aggregate")] + seeds(XG, "/per_seed/{s}/r4_comp_nway/mean_novel_per_corner/alone:pca_tensor"),
}

# numbers the scanner does not see: (draft anchor, verbatim draft string, artefact, pointer-or-verbatim)
EXTRA = [
    ("PoE (own budget, not matched) & 3 & 0 / 0 / 0", "0 / 0 / 0", G, [R34 + f"/arms/PoE/per_seed/{s}/row3_pair" for s in range(3)]),
    ("PoE (own budget, not matched) & 3 & 0 / 0 / 0", "0.020 / 0 / 0", G, [R34 + f"/arms/PoE/per_seed/{s}/row4_nway" for s in range(3)]),
    ("the store wins 109/109 corners against each, on every seed. Scope", "109/109", G,
     [R34 + f"/paired_vs_kstar/{a}/{s}/corners_control_wins" for a in ("LACE", "PoE") for s in range(3)]),
    ("0.000 / 0.000, and the store wins 109/109 corners", "109/109", G,
     [R34 + f"/paired_vs_kstar/{a}/{s}/corners_control_wins" for a in ("LACE", "PoE") for s in range(3)]),
    ("(60 of 60 at $n=2$, 120 of 120 at $n\\ge3$)", "60 of 60", G,
     [f"/PART_B_ladder/k=2/paired_vs_kstar/{a}/{s}/corners_control_wins" for a in ("LACE", "PoE") for s in range(3)]),
    ("(60 of 60 at $n=2$, 120 of 120 at $n\\ge3$)", "120 of 120", G,
     [f"/PART_B_ladder/k={k}/paired_vs_kstar/{a}/{s}/corners_control_wins" for k in range(3, 7) for a in ("LACE", "PoE") for s in range(3)]),
    ("negative in only 2 of 6 cells, while $A''$ is positive in 6 of 6 (its interval excludes 0 in 5 of", "5 of", XG,
     [f"/per_seed/{s}/{r}/encoder_comparison/BAR_Adprime_swap_loss_alone/bootstrap_ci95/0" for r in ("r3_comp_2way", "r4_comp_nway") for s in range(3)]),
    ("about +32 tuples per corner on row 4 at seeds 0 and 2", "+32", XG,
     [f"/per_seed/{s}/r4_comp_nway/encoder_comparison/BAR_Aprime_loop_nosw_minus_alone/mean_paired_delta" for s in (0, 2)]),
]
