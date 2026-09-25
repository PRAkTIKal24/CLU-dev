## 13. ⛔⛔ PASS 5: THE w15 FOLD, CLOSED. EVERY w15 NUMBER THE DRAFTS PRINT, CHECKED THREE WAYS

Passes 3 and 4 declared this fold out of brief (§6.1 stayed STALE). It is in brief now, and it is **closed**: the drafts print these numbers, and from this pass the register checks every one of them.

**Method (`w22_seeds5/w15_check.py`, `w15_map.py`).** Every number token that either draft prints under a `% SRC:` line or `% SRCBLOCK:` naming `w15-bar-d-score`, `w15-coverage-control` or `w15-lace-poe-rows` is found by `scan_draft.py` (decimals, signed decimals, comma-grouped integers, integers of four or more digits, and `a of b` / `a/b` counts). Each gets three cells in the rows below:
1. **the draft**: the value, verbatim on its line of the pinned pass-5 copy of each draft (`draft_pass5/`, the installed bytes);
2. **the record the draft cites**: verbatim on a line of that spoke's `FINDINGS.md` (the provenance comment's own claim);
3. **the artefact** (a paraphrase is not a source): the JSON leaf named by an RFC-6901 pointer, and the build re-checks that the printed value **is** that leaf at the printed precision (exact, or rounded half-up / half-even from the float's repr). A count is checked as integer equality. Where the artefact is plain text (`IDENTITY-CHECK.txt`), the cell is verbatim on its line.

Numbers the token scanner cannot see (integer zeros inside a table cell, counts that wrap a line) are listed after each spoke's table, each against its JSON leaves.

### 13.1 `coverage-control` → `tab:ladder`'s coverage row and the coverage-control paragraph: 27 numbers

| # | `DRAFT-v2.tex` | `DRAFT-v2-style.tex` | the cited `FINDINGS.md` | the artefact (JSON leaf at the printed precision, or text line) |
|---|---|---|---|---|
| 1 | {{p5_w15d_1}} | {{p5_w15s_1}} | {{p5_w15f_1}} | {{p5_w15j_1}} (exact) |
| 2 | {{p5_w15d_2}} | {{p5_w15s_2}} | {{p5_w15f_2}} | {{p5_w15j_2}} (half-up) |
| 3 | {{p5_w15d_3}} | {{p5_w15s_3}} | {{p5_w15f_3}} | {{p5_w15j_3}} (half-up) |
| 4 | {{p5_w15d_4}} | {{p5_w15s_4}} | {{p5_w15f_4}} | {{p5_w15j_4}} (half-up) |
| 5 | {{p5_w15d_5}} | {{p5_w15s_5}} | {{p5_w15f_5}} | {{p5_w15j_5}} (half-up) |
| 6 | {{p5_w15d_6}} | {{p5_w15s_6}} | {{p5_w15f_6}} | {{p5_w15j_6}} (half-up) |
| 7 | {{p5_w15d_7}} | {{p5_w15s_7}} | {{p5_w15f_7}} | {{p5_w15j_7}} (half-up) |
| 8 | {{p5_w15d_8}} | {{p5_w15s_8}} | {{p5_w15f_8}} | {{p5_w15j_8}} (half-up) |
| 9 | {{p5_w15d_9}} | {{p5_w15s_9}} | {{p5_w15f_9}} | {{p5_w15j_9}} (half-up) |
| 10 | {{p5_w15d_10}} | {{p5_w15s_10}} | {{p5_w15f_10}} | {{p5_w15t_1}} |
| 11 | {{p5_w15d_11}} | {{p5_w15s_11}} | {{p5_w15f_11}} | {{p5_w15t_2}} |
| 12 | {{p5_w15d_12}} | {{p5_w15s_12}} | {{p5_w15f_12}} | {{p5_w15t_3}} |
| 13 | {{p5_w15d_13}} | {{p5_w15s_13}} | {{p5_w15f_13}} | {{p5_w15t_4}} |
| 14 | {{p5_w15d_14}} | {{p5_w15s_14}} | {{p5_w15f_14}} | {{p5_w15j_10}} (half-up) |
| 15 | {{p5_w15d_15}} | {{p5_w15s_15}} | {{p5_w15f_15}} | {{p5_w15j_11}} (half-up) |
| 16 | {{p5_w15d_16}} | {{p5_w15s_16}} | {{p5_w15f_16}} | {{p5_w15j_12}} (exact) |
| 17 | {{p5_w15d_17}} | {{p5_w15s_17}} | {{p5_w15f_17}} | {{p5_w15j_13}} (half-up) |
| 18 | {{p5_w15d_18}} | {{p5_w15s_18}} | {{p5_w15f_18}} | {{p5_w15j_14}} (half-up) |
| 19 | {{p5_w15d_19}} | {{p5_w15s_19}} | {{p5_w15f_19}} | {{p5_w15j_15}} (half-up) |
| 20 | {{p5_w15d_20}} | {{p5_w15s_20}} | {{p5_w15f_20}} | {{p5_w15j_16}} (half-up) |
| 21 | {{p5_w15d_21}} | {{p5_w15s_21}} | {{p5_w15f_21}} | {{p5_w15j_17}} (exact) |
| 22 | {{p5_w15d_22}} | {{p5_w15s_22}} | {{p5_w15f_22}} | {{p5_w15j_18}} (exact) |
| 23 | {{p5_w15d_23}} | {{p5_w15s_23}} | {{p5_w15f_23}} | {{p5_w15j_19}} (exact) |
| 24 | {{p5_w15d_24}} | {{p5_w15s_24}} | {{p5_w15f_24}} | {{p5_w15j_20}} (half-up) |
| 25 | {{p5_w15d_25}} | {{p5_w15s_25}} | {{p5_w15f_25}} | {{p5_w15j_21}} (half-up) |
| 26 | {{p5_w15d_26}} | {{p5_w15s_26}} | {{p5_w15f_26}} | {{p5_w15j_22}} (half-up) |
| 27 | {{p5_w15d_27}} | {{p5_w15s_27}} | {{p5_w15f_27}} | {{p5_w15j_23}} (exact) |

### 13.2 `lace-poe-rows` → `tab:gentable-kstar`, `tab:lace-poe` (both blocks), the LACE/PoE paragraphs: 144 numbers

| # | `DRAFT-v2.tex` | `DRAFT-v2-style.tex` | the cited `FINDINGS.md` | the artefact (JSON leaf at the printed precision, or text line) |
|---|---|---|---|---|
| 1 | {{p5_w15d_28}} | {{p5_w15s_28}} | {{p5_w15f_28}} | {{p5_w15j_24}} (half-up) |
| 2 | {{p5_w15d_29}} | {{p5_w15s_29}} | {{p5_w15f_29}} | {{p5_w15j_25}} (half-up) |
| 3 | {{p5_w15d_30}} | {{p5_w15s_30}} | {{p5_w15f_30}} | {{p5_w15j_26}} (exact) |
| 4 | {{p5_w15d_31}} | {{p5_w15s_31}} | {{p5_w15f_31}} | {{p5_w15j_27}} (half-up) |
| 5 | {{p5_w15d_32}} | {{p5_w15s_32}} | {{p5_w15f_32}} | {{p5_w15j_28}} (half-up) |
| 6 | {{p5_w15d_33}} | {{p5_w15s_33}} | {{p5_w15f_33}} | {{p5_w15j_29}} (half-up) |
| 7 | {{p5_w15d_34}} | {{p5_w15s_34}} | {{p5_w15f_34}} | {{p5_w15j_30}} (half-up) |
| 8 | {{p5_w15d_35}} | {{p5_w15s_35}} | {{p5_w15f_35}} | {{p5_w15j_31}} (half-up) |
| 9 | {{p5_w15d_36}} | {{p5_w15s_36}} | {{p5_w15f_36}} | {{p5_w15j_32}} (half-up) |
| 10 | {{p5_w15d_37}} | {{p5_w15s_37}} | {{p5_w15f_37}} | {{p5_w15j_33}} (half-up) |
| 11 | {{p5_w15d_38}} | {{p5_w15s_38}} | {{p5_w15f_38}} | {{p5_w15j_34}} (half-up) |
| 12 | {{p5_w15d_39}} | {{p5_w15s_39}} | {{p5_w15f_39}} | {{p5_w15j_35}} (half-up) |
| 13 | {{p5_w15d_40}} | {{p5_w15s_40}} | {{p5_w15f_40}} | {{p5_w15j_36}} (half-up) |
| 14 | {{p5_w15d_41}} | {{p5_w15s_41}} | {{p5_w15f_41}} | {{p5_w15j_37}} (exact) |
| 15 | {{p5_w15d_42}} | {{p5_w15s_42}} | {{p5_w15f_42}} | {{p5_w15j_38}} (half-up) |
| 16 | {{p5_w15d_43}} | {{p5_w15s_43}} | {{p5_w15f_43}} | {{p5_w15j_39}} (exact) |
| 17 | {{p5_w15d_44}} | {{p5_w15s_44}} | {{p5_w15f_44}} | {{p5_w15j_40}} (exact) |
| 18 | {{p5_w15d_45}} | {{p5_w15s_45}} | {{p5_w15f_45}} | {{p5_w15j_41}} (exact) |
| 19 | {{p5_w15d_46}} | {{p5_w15s_46}} | {{p5_w15f_46}} | {{p5_w15j_42}} (half-up) |
| 20 | {{p5_w15d_47}} | {{p5_w15s_47}} | {{p5_w15f_47}} | {{p5_w15j_43}} (half-up) |
| 21 | {{p5_w15d_48}} | {{p5_w15s_48}} | {{p5_w15f_48}} | {{p5_w15j_44}} (half-up) |
| 22 | {{p5_w15d_49}} | {{p5_w15s_49}} | {{p5_w15f_49}} | {{p5_w15j_45}} (half-up) |
| 23 | {{p5_w15d_50}} | {{p5_w15s_50}} | {{p5_w15f_50}} | {{p5_w15j_46}} (half-up) |
| 24 | {{p5_w15d_51}} | {{p5_w15s_51}} | {{p5_w15f_51}} | {{p5_w15j_47}} (half-up) |
| 25 | {{p5_w15d_52}} | {{p5_w15s_52}} | {{p5_w15f_52}} | {{p5_w15j_48}} (half-up) |
| 26 | {{p5_w15d_53}} | {{p5_w15s_53}} | {{p5_w15f_53}} | {{p5_w15j_49}} (half-up) |
| 27 | {{p5_w15d_54}} | {{p5_w15s_54}} | {{p5_w15f_54}} | {{p5_w15j_50}} (half-up) |
| 28 | {{p5_w15d_55}} | {{p5_w15s_55}} | {{p5_w15f_55}} | {{p5_w15j_51}} (half-up) |
| 29 | {{p5_w15d_56}} | {{p5_w15s_56}} | {{p5_w15f_56}} | {{p5_w15j_52}} (half-up) |
| 30 | {{p5_w15d_57}} | {{p5_w15s_57}} | {{p5_w15f_57}} | {{p5_w15j_53}} (half-up) |
| 31 | {{p5_w15d_58}} | {{p5_w15s_58}} | {{p5_w15f_58}} | {{p5_w15j_54}} (half-up) |
| 32 | {{p5_w15d_59}} | {{p5_w15s_59}} | {{p5_w15f_59}} | {{p5_w15j_55}} (half-up) |
| 33 | {{p5_w15d_60}} | {{p5_w15s_60}} | {{p5_w15f_60}} | {{p5_w15j_56}} (half-up) |
| 34 | {{p5_w15d_61}} | {{p5_w15s_61}} | {{p5_w15f_61}} | {{p5_w15j_57}} (half-up) |
| 35 | {{p5_w15d_62}} | {{p5_w15s_62}} | {{p5_w15f_62}} | {{p5_w15j_58}} (half-up) |
| 36 | {{p5_w15d_63}} | {{p5_w15s_63}} | {{p5_w15f_63}} | {{p5_w15j_59}} (half-up) |
| 37 | {{p5_w15d_64}} | {{p5_w15s_64}} | {{p5_w15f_64}} | {{p5_w15j_60}} (half-up) |
| 38 | {{p5_w15d_65}} | {{p5_w15s_65}} | {{p5_w15f_65}} | {{p5_w15j_61}} (half-up) |
| 39 | {{p5_w15d_66}} | {{p5_w15s_66}} | {{p5_w15f_66}} | {{p5_w15j_62}} (half-up) |
| 40 | {{p5_w15d_67}} | {{p5_w15s_67}} | {{p5_w15f_67}} | {{p5_w15j_63}} (half-up) |
| 41 | {{p5_w15d_68}} | {{p5_w15s_68}} | {{p5_w15f_68}} | {{p5_w15j_64}} (half-up) |
| 42 | {{p5_w15d_69}} | {{p5_w15s_69}} | {{p5_w15f_69}} | {{p5_w15j_65}} (exact) |
| 43 | {{p5_w15d_70}} | {{p5_w15s_70}} | {{p5_w15f_70}} | {{p5_w15j_66}} (half-up) |
| 44 | {{p5_w15d_71}} | {{p5_w15s_71}} | {{p5_w15f_71}} | {{p5_w15j_67}} (half-up) |
| 45 | {{p5_w15d_72}} | {{p5_w15s_72}} | {{p5_w15f_72}} | {{p5_w15j_68}} (half-up) |
| 46 | {{p5_w15d_73}} | {{p5_w15s_73}} | {{p5_w15f_73}} | {{p5_w15j_69}} (half-up) |
| 47 | {{p5_w15d_74}} | {{p5_w15s_74}} | {{p5_w15f_74}} | {{p5_w15j_70}} (exact) |
| 48 | {{p5_w15d_75}} | {{p5_w15s_75}} | {{p5_w15f_75}} | {{p5_w15j_71}} (half-up) |
| 49 | {{p5_w15d_76}} | {{p5_w15s_76}} | {{p5_w15f_76}} | {{p5_w15j_72}} (half-up) |
| 50 | {{p5_w15d_77}} | {{p5_w15s_77}} | {{p5_w15f_77}} | {{p5_w15j_73}} (half-up) |
| 51 | {{p5_w15d_78}} | {{p5_w15s_78}} | {{p5_w15f_78}} | {{p5_w15j_74}} (half-up) |
| 52 | {{p5_w15d_79}} | {{p5_w15s_79}} | {{p5_w15f_79}} | {{p5_w15j_75}} (half-up) |
| 53 | {{p5_w15d_80}} | {{p5_w15s_80}} | {{p5_w15f_80}} | {{p5_w15j_76}} (exact) |
| 54 | {{p5_w15d_81}} | {{p5_w15s_81}} | {{p5_w15f_81}} | {{p5_w15j_77}} (half-up) |
| 55 | {{p5_w15d_82}} | {{p5_w15s_82}} | {{p5_w15f_82}} | {{p5_w15j_78}} (half-up) |
| 56 | {{p5_w15d_83}} | {{p5_w15s_83}} | {{p5_w15f_83}} | {{p5_w15j_79}} (half-up) |
| 57 | {{p5_w15d_84}} | {{p5_w15s_84}} | {{p5_w15f_84}} | {{p5_w15j_80}} (half-up) |
| 58 | {{p5_w15d_85}} | {{p5_w15s_85}} | {{p5_w15f_85}} | {{p5_w15j_81}} (half-up) |
| 59 | {{p5_w15d_86}} | {{p5_w15s_86}} | {{p5_w15f_86}} | {{p5_w15j_82}} (half-up) |
| 60 | {{p5_w15d_87}} | {{p5_w15s_87}} | {{p5_w15f_87}} | {{p5_w15j_83}} (half-up) |
| 61 | {{p5_w15d_88}} | {{p5_w15s_88}} | {{p5_w15f_88}} | {{p5_w15j_84}} (half-up) |
| 62 | {{p5_w15d_89}} | {{p5_w15s_89}} | {{p5_w15f_89}} | {{p5_w15j_85}} (half-up) |
| 63 | {{p5_w15d_90}} | {{p5_w15s_90}} | {{p5_w15f_90}} | {{p5_w15j_86}} (half-up) |
| 64 | {{p5_w15d_91}} | {{p5_w15s_91}} | {{p5_w15f_91}} | {{p5_w15j_87}} (exact) |
| 65 | {{p5_w15d_92}} | {{p5_w15s_92}} | {{p5_w15f_92}} | {{p5_w15j_88}} (half-up) |
| 66 | {{p5_w15d_93}} | {{p5_w15s_93}} | {{p5_w15f_93}} | {{p5_w15j_89}} (half-up) |
| 67 | {{p5_w15d_94}} | {{p5_w15s_94}} | {{p5_w15f_94}} | {{p5_w15j_90}} (half-up) |
| 68 | {{p5_w15d_95}} | {{p5_w15s_95}} | {{p5_w15f_95}} | {{p5_w15j_91}} (half-up) |
| 69 | {{p5_w15d_96}} | {{p5_w15s_96}} | {{p5_w15f_96}} | {{p5_w15j_92}} (half-up) |
| 70 | {{p5_w15d_97}} | {{p5_w15s_97}} | {{p5_w15f_97}} | {{p5_w15j_93}} (half-up) |
| 71 | {{p5_w15d_98}} | {{p5_w15s_98}} | {{p5_w15f_98}} | {{p5_w15j_94}} (half-up) |
| 72 | {{p5_w15d_99}} | {{p5_w15s_99}} | {{p5_w15f_99}} | {{p5_w15j_95}} (half-up) |
| 73 | {{p5_w15d_100}} | {{p5_w15s_100}} | {{p5_w15f_100}} | {{p5_w15j_96}} (half-up) |
| 74 | {{p5_w15d_101}} | {{p5_w15s_101}} | {{p5_w15f_101}} | {{p5_w15j_97}} (half-up) |
| 75 | {{p5_w15d_102}} | {{p5_w15s_102}} | {{p5_w15f_102}} | {{p5_w15j_98}} (half-up) |
| 76 | {{p5_w15d_103}} | {{p5_w15s_103}} | {{p5_w15f_103}} | {{p5_w15j_99}} (half-up) |
| 77 | {{p5_w15d_104}} | {{p5_w15s_104}} | {{p5_w15f_104}} | {{p5_w15j_100}} (half-up) |
| 78 | {{p5_w15d_105}} | {{p5_w15s_105}} | {{p5_w15f_105}} | {{p5_w15j_101}} (half-up) |
| 79 | {{p5_w15d_106}} | {{p5_w15s_106}} | {{p5_w15f_106}} | {{p5_w15j_102}} (half-up) |
| 80 | {{p5_w15d_107}} | {{p5_w15s_107}} | {{p5_w15f_107}} | {{p5_w15j_103}} (half-up) |
| 81 | {{p5_w15d_108}} | {{p5_w15s_108}} | {{p5_w15f_108}} | {{p5_w15j_104}} (half-up) |
| 82 | {{p5_w15d_109}} | {{p5_w15s_109}} | {{p5_w15f_109}} | {{p5_w15j_105}} (half-up) |
| 83 | {{p5_w15d_110}} | {{p5_w15s_110}} | {{p5_w15f_110}} | {{p5_w15j_106}} (half-up) |
| 84 | {{p5_w15d_111}} | {{p5_w15s_111}} | {{p5_w15f_111}} | {{p5_w15j_107}} (half-up) |
| 85 | {{p5_w15d_112}} | {{p5_w15s_112}} | {{p5_w15f_112}} | {{p5_w15j_108}} (half-up) |
| 86 | {{p5_w15d_113}} | {{p5_w15s_113}} | {{p5_w15f_113}} | {{p5_w15j_109}} (half-up) |
| 87 | {{p5_w15d_114}} | {{p5_w15s_114}} | {{p5_w15f_114}} | {{p5_w15j_110}} (half-up) |
| 88 | {{p5_w15d_115}} | {{p5_w15s_115}} | {{p5_w15f_115}} | {{p5_w15j_111}} (half-up) |
| 89 | {{p5_w15d_116}} | {{p5_w15s_116}} | {{p5_w15f_116}} | {{p5_w15j_112}} (half-up) |
| 90 | {{p5_w15d_117}} | {{p5_w15s_117}} | {{p5_w15f_117}} | {{p5_w15j_113}} (half-up) |
| 91 | {{p5_w15d_118}} | {{p5_w15s_118}} | {{p5_w15f_118}} | {{p5_w15j_114}} (half-up) |
| 92 | {{p5_w15d_119}} | {{p5_w15s_119}} | {{p5_w15f_119}} | {{p5_w15j_115}} (exact) |
| 93 | {{p5_w15d_120}} | {{p5_w15s_120}} | not in FINDINGS, which prints the truncated value; the draft line now also cites the JSON (§13.4) | {{p5_w15j_116}} (exact) |
| 94 | {{p5_w15d_121}} | {{p5_w15s_121}} | {{p5_w15f_120}} | {{p5_w15j_117}} (exact) |
| 95 | {{p5_w15d_122}} | {{p5_w15s_122}} | {{p5_w15f_121}} | a count: checked against its JSON leaves in the list below this table |
| 96 | {{p5_w15d_123}} | {{p5_w15s_123}} | {{p5_w15f_122}} | {{p5_w15j_118}} (half-up) |
| 97 | {{p5_w15d_124}} | {{p5_w15s_124}} | {{p5_w15f_123}} | {{p5_w15j_119}} (exact) |
| 98 | {{p5_w15d_125}} | {{p5_w15s_125}} | {{p5_w15f_124}} | {{p5_w15j_120}} (half-up) |
| 99 | {{p5_w15d_126}} | {{p5_w15s_126}} | {{p5_w15f_125}} | {{p5_w15j_121}} (half-up) |
| 100 | {{p5_w15d_127}} | {{p5_w15s_127}} | {{p5_w15f_126}} | {{p5_w15j_122}} (exact) |
| 101 | {{p5_w15d_128}} | {{p5_w15s_128}} | {{p5_w15f_127}} | {{p5_w15j_123}} (half-up) |
| 102 | {{p5_w15d_129}} | {{p5_w15s_129}} | {{p5_w15f_128}} | {{p5_w15j_124}} (half-up) |
| 103 | {{p5_w15d_130}} | {{p5_w15s_130}} | {{p5_w15f_129}} | {{p5_w15j_125}} (half-up) |
| 104 | {{p5_w15d_131}} | {{p5_w15s_131}} | {{p5_w15f_130}} | {{p5_w15j_126}} (half-up) |
| 105 | {{p5_w15d_132}} | {{p5_w15s_132}} | {{p5_w15f_131}} | {{p5_w15j_127}} (half-up) |
| 106 | {{p5_w15d_133}} | {{p5_w15s_133}} | {{p5_w15f_132}} | {{p5_w15j_128}} (half-up) |
| 107 | {{p5_w15d_134}} | {{p5_w15s_134}} | {{p5_w15f_133}} | {{p5_w15j_129}} (half-up) |
| 108 | {{p5_w15d_135}} | {{p5_w15s_135}} | {{p5_w15f_134}} | {{p5_w15j_130}} (exact) |
| 109 | {{p5_w15d_136}} | {{p5_w15s_136}} | {{p5_w15f_135}} | {{p5_w15j_131}} (exact) |
| 110 | {{p5_w15d_137}} | {{p5_w15s_137}} | {{p5_w15f_136}} | {{p5_w15j_132}} (exact) |
| 111 | {{p5_w15d_138}} | {{p5_w15s_138}} | {{p5_w15f_137}} | {{p5_w15j_133}} (half-up) |
| 112 | {{p5_w15d_139}} | {{p5_w15s_139}} | {{p5_w15f_138}} | {{p5_w15j_134}} (exact) |
| 113 | {{p5_w15d_140}} | {{p5_w15s_140}} | {{p5_w15f_139}} | {{p5_w15j_135}} (exact) |
| 114 | {{p5_w15d_141}} | {{p5_w15s_141}} | {{p5_w15f_140}} | {{p5_w15j_136}} (exact) |
| 115 | {{p5_w15d_142}} | {{p5_w15s_142}} | {{p5_w15f_141}} | {{p5_w15j_137}} (exact) |
| 116 | {{p5_w15d_143}} | {{p5_w15s_143}} | {{p5_w15f_142}} | {{p5_w15j_138}} (exact) |
| 117 | {{p5_w15d_144}} | {{p5_w15s_144}} | {{p5_w15f_143}} | {{p5_w15j_139}} (half-up) |
| 118 | {{p5_w15d_145}} | {{p5_w15s_145}} | {{p5_w15f_144}} | {{p5_w15j_140}} (exact) |
| 119 | {{p5_w15d_146}} | {{p5_w15s_146}} | {{p5_w15f_145}} | {{p5_w15j_141}} (half-up) |
| 120 | {{p5_w15d_147}} | {{p5_w15s_147}} | {{p5_w15f_146}} | {{p5_w15j_142}} (exact) |
| 121 | {{p5_w15d_148}} | {{p5_w15s_148}} | {{p5_w15f_147}} | {{p5_w15j_143}} (half-up) |
| 122 | {{p5_w15d_149}} | {{p5_w15s_149}} | {{p5_w15f_148}} | {{p5_w15j_144}} (exact) |
| 123 | {{p5_w15d_150}} | {{p5_w15s_150}} | {{p5_w15f_149}} | {{p5_w15j_145}} (exact) |
| 124 | {{p5_w15d_151}} | {{p5_w15s_151}} | {{p5_w15f_150}} | {{p5_w15j_146}} (half-up) |
| 125 | {{p5_w15d_152}} | {{p5_w15s_152}} | {{p5_w15f_151}} | {{p5_w15j_147}} (exact) |
| 126 | {{p5_w15d_153}} | {{p5_w15s_153}} | {{p5_w15f_152}} | {{p5_w15j_148}} (half-up) |
| 127 | {{p5_w15d_154}} | {{p5_w15s_154}} | {{p5_w15f_153}} | {{p5_w15j_149}} (half-up) |
| 128 | {{p5_w15d_155}} | {{p5_w15s_155}} | {{p5_w15f_154}} | {{p5_w15j_150}} (half-up) |
| 129 | {{p5_w15d_156}} | {{p5_w15s_156}} | {{p5_w15f_155}} | {{p5_w15j_151}} (half-up) |
| 130 | {{p5_w15d_157}} | {{p5_w15s_157}} | {{p5_w15f_156}} | {{p5_w15j_152}} (half-up) |
| 131 | {{p5_w15d_158}} | {{p5_w15s_158}} | {{p5_w15f_157}} | {{p5_w15j_153}} (half-up) |
| 132 | {{p5_w15d_159}} | {{p5_w15s_159}} | {{p5_w15f_158}} | {{p5_w15j_154}} (half-up) |
| 133 | {{p5_w15d_160}} | {{p5_w15s_160}} | {{p5_w15f_159}} | {{p5_w15j_155}} (exact) |
| 134 | {{p5_w15d_161}} | {{p5_w15s_161}} | {{p5_w15f_160}} | {{p5_w15j_156}} (exact) |
| 135 | {{p5_w15d_162}} | {{p5_w15s_162}} | {{p5_w15f_161}} | a count: checked against its JSON leaves in the list below this table |
| 136 | {{p5_w15d_163}} | {{p5_w15s_163}} | {{p5_w15f_162}} | a count: checked against its JSON leaves in the list below this table |
| 137 | {{p5_w15d_164}} | {{p5_w15s_164}} | {{p5_w15f_163}} | a count: checked against its JSON leaves in the list below this table |
| 138 | {{p5_w15d_165}} | {{p5_w15s_165}} | {{p5_w15f_164}} | {{p5_w15j_157}} (exact) |
| 139 | {{p5_w15d_166}} | {{p5_w15s_166}} | {{p5_w15f_165}} | {{p5_w15j_158}} (exact) |
| 140 | {{p5_w15d_167}} | {{p5_w15s_167}} | {{p5_w15f_166}} | {{p5_w15j_159}} (exact) |
| 141 | {{p5_w15d_168}} | {{p5_w15s_168}} | {{p5_w15f_167}} | {{p5_w15j_160}} (half-up) |
| 142 | {{p5_w15d_169}} | {{p5_w15s_169}} | {{p5_w15f_168}} | {{p5_w15j_161}} (exact) |
| 143 | {{p5_w15d_170}} | {{p5_w15s_170}} | not in FINDINGS, which prints the truncated value; the draft line now also cites the JSON (§13.4) | {{p5_w15j_162}} (exact) |
| 144 | {{p5_w15d_171}} | {{p5_w15s_171}} | {{p5_w15f_169}} | {{p5_w15j_163}} (exact) |

Numbers the scanner cannot see (w15-lace-poe-rows):

- {{p5_w15ed_1}} and {{p5_w15es_1}}: the JSON leaves it reads (3): {{p5_w15e_1}} · {{p5_w15e_2}} · {{p5_w15e_3}}
- {{p5_w15ed_2}} and {{p5_w15es_2}}: the JSON leaves it reads (3): {{p5_w15e_4}} · {{p5_w15e_5}} · {{p5_w15e_6}}
- {{p5_w15ed_3}} and {{p5_w15es_3}}: the JSON leaves it reads (6): {{p5_w15e_7}} · {{p5_w15e_8}} · {{p5_w15e_9}} · {{p5_w15e_10}} · {{p5_w15e_11}} · {{p5_w15e_12}}
- {{p5_w15ed_4}} and {{p5_w15es_4}}: the JSON leaves it reads (6): {{p5_w15e_13}} · {{p5_w15e_14}} · {{p5_w15e_15}} · {{p5_w15e_16}} · {{p5_w15e_17}} · {{p5_w15e_18}}
- {{p5_w15ed_5}} and {{p5_w15es_5}}: the JSON leaves it reads (6): {{p5_w15e_19}} · {{p5_w15e_20}} · {{p5_w15e_21}} · {{p5_w15e_22}} · {{p5_w15e_23}} · {{p5_w15e_24}}
- {{p5_w15ed_6}} and {{p5_w15es_6}}: the JSON leaves it reads (24): {{p5_w15e_25}} · {{p5_w15e_26}} · {{p5_w15e_27}} · {{p5_w15e_28}} · {{p5_w15e_29}} · {{p5_w15e_30}} · {{p5_w15e_31}} · {{p5_w15e_32}} · {{p5_w15e_33}} · {{p5_w15e_34}} · {{p5_w15e_35}} · {{p5_w15e_36}} · {{p5_w15e_37}} · {{p5_w15e_38}} · {{p5_w15e_39}} · {{p5_w15e_40}} · {{p5_w15e_41}} · {{p5_w15e_42}} · {{p5_w15e_43}} · {{p5_w15e_44}} · {{p5_w15e_45}} · {{p5_w15e_46}} · {{p5_w15e_47}} · {{p5_w15e_48}}

### 13.3 `bar-d-score` → the Criterion D line, `tab:bard-gpu`, the decomposition paragraph: 28 numbers

| # | `DRAFT-v2.tex` | `DRAFT-v2-style.tex` | the cited `FINDINGS.md` | the artefact (JSON leaf at the printed precision, or text line) |
|---|---|---|---|---|
| 1 | {{p5_w15d_172}} | {{p5_w15s_172}} | {{p5_w15f_170}} | {{p5_w15j_164}} (exact) |
| 2 | {{p5_w15d_173}} | {{p5_w15s_173}} | {{p5_w15f_171}} | {{p5_w15j_165}} (exact) |
| 3 | {{p5_w15d_174}} | {{p5_w15s_174}} | {{p5_w15f_172}} | {{p5_w15j_166}} (exact) |
| 4 | {{p5_w15d_175}} | {{p5_w15s_175}} | {{p5_w15f_173}} | {{p5_w15j_167}} (exact) |
| 5 | {{p5_w15d_176}} | {{p5_w15s_176}} | {{p5_w15f_174}} | {{p5_w15j_168}} (exact) |
| 6 | {{p5_w15d_177}} | {{p5_w15s_177}} | {{p5_w15f_175}} | {{p5_w15j_169}} (half-up) |
| 7 | {{p5_w15d_178}} | {{p5_w15s_178}} | {{p5_w15f_176}} | {{p5_w15j_170}} (half-up) |
| 8 | {{p5_w15d_179}} | {{p5_w15s_179}} | {{p5_w15f_177}} | {{p5_w15j_171}} (exact) |
| 9 | {{p5_w15d_180}} | {{p5_w15s_180}} | {{p5_w15f_178}} | {{p5_w15j_172}} (half-up) |
| 10 | {{p5_w15d_181}} | {{p5_w15s_181}} | {{p5_w15f_179}} | {{p5_w15j_173}} (half-up) |
| 11 | {{p5_w15d_182}} | {{p5_w15s_182}} | {{p5_w15f_180}} | {{p5_w15j_174}} (half-up) |
| 12 | {{p5_w15d_183}} | {{p5_w15s_183}} | {{p5_w15f_181}} | {{p5_w15j_175}} (exact) |
| 13 | {{p5_w15d_184}} | {{p5_w15s_184}} | {{p5_w15f_182}} | {{p5_w15j_176}} (exact) |
| 14 | {{p5_w15d_185}} | {{p5_w15s_185}} | {{p5_w15f_183}} | {{p5_w15j_177}} (half-up) |
| 15 | {{p5_w15d_186}} | {{p5_w15s_186}} | {{p5_w15f_184}} | {{p5_w15j_178}} (half-up) |
| 16 | {{p5_w15d_187}} | {{p5_w15s_187}} | {{p5_w15f_185}} | {{p5_w15j_179}} (half-up) |
| 17 | {{p5_w15d_188}} | {{p5_w15s_188}} | {{p5_w15f_186}} | {{p5_w15j_180}} (half-up) |
| 18 | {{p5_w15d_189}} | {{p5_w15s_189}} | {{p5_w15f_187}} | {{p5_w15j_181}} (count-equal) |
| 19 | {{p5_w15d_190}} | {{p5_w15s_190}} | {{p5_w15f_188}} | {{p5_w15j_182}} (count-equal) |
| 20 | {{p5_w15d_191}} | {{p5_w15s_191}} | {{p5_w15f_189}} | {{p5_w15j_183}} (count-equal) |
| 21 | {{p5_w15d_192}} | {{p5_w15s_192}} | {{p5_w15f_190}} | {{p5_w15j_184}} (count-equal) |
| 22 | {{p5_w15d_193}} | {{p5_w15s_193}} | {{p5_w15f_191}} | {{p5_w15j_185}} (count-equal) |
| 23 | {{p5_w15d_194}} | {{p5_w15s_194}} | {{p5_w15f_192}} | {{p5_w15j_186}} (half-up) |
| 24 | {{p5_w15d_195}} | {{p5_w15s_195}} | {{p5_w15f_193}} | {{p5_w15j_187}} (half-up) |
| 25 | {{p5_w15d_196}} | {{p5_w15s_196}} | {{p5_w15f_194}} | {{p5_w15j_188}} (half-up) |
| 26 | {{p5_w15d_197}} | {{p5_w15s_197}} | {{p5_w15f_195}} | {{p5_w15j_189}} (half-up) |
| 27 | {{p5_w15d_198}} | {{p5_w15s_198}} | {{p5_w15f_196}} | {{p5_w15j_190}} (half-up) |
| 28 | {{p5_w15d_199}} | {{p5_w15s_199}} | {{p5_w15f_197}} | {{p5_w15j_191}} (half-up) |

Numbers the scanner cannot see (w15-bar-d-score):

- {{p5_w15ed_7}} and {{p5_w15es_7}}: the JSON leaves it reads (6): {{p5_w15e_49}} · {{p5_w15e_50}} · {{p5_w15e_51}} · {{p5_w15e_52}} · {{p5_w15e_53}} · {{p5_w15e_54}}
- {{p5_w15ed_8}} and {{p5_w15es_8}}: the JSON leaves it reads (2): {{p5_w15e_55}} · {{p5_w15e_56}}

### 13.4 ⛔ The one disagreement, and what was done

The pre-pass-5 drafts printed PoE's budget as {{p5_w15q_1}} the reference, in two places per draft, copying the spoke's `FINDINGS.md`. The acceptance JSON holds {{p5_w15q_2}}, which is {{p5_w15q_3}} B over the {{p5_w15q_4}} B reference. Rounded to two decimals, that ratio is one hundredth above what FINDINGS prints: FINDINGS truncated where it should have rounded. **The artefact wins.** Both drafts now print the JSON's own value, {{p5_w15q_5}}×, with `GENTABLE-KSTAR.json` added to the line's `% SRC:` (`w22_seeds5/apply_pass5_edits.py`, edit E7). The rows above check the corrected value.

**Result.** 199 numbers per draft, each verbatim in both drafts and in its cited `FINDINGS.md`, and each equal to its artefact (199 of 199; the extra cells cover the counts and zeros). **Nothing in the three spokes is left unfolded.** One value quoted by `bar-d-score` is not its own: the `alone` probe margins are w8's, and they are checked against the w8 probe JSON that `bar-d-score`'s `FINDINGS.md` names. Where a draft line cites FINDINGS for a number the JSON holds at more digits, the draft's rounding is the one printed in the rows above (half-up or half-even).

## 14. WAVE 20 `hub/shapes-data-matched` (3 of 3 scored runs spent). FOLDED from `SHAPES-MATCHED.json` and its rendered tables

§11 said "awaiting". It has landed: `evidence/outputs/w20-shapes-data-matched/`. Rows are copied verbatim from `SHAPES-MATCHED.md` (rendered from the JSON by the hub); the headline is also checked against the JSON leaf.

### 14.1 The headline: the data-matched margin on rows 3–4, Shapes3D, K\* = 32, TEST, 3 seeds

- margin, mean: {{p5_w20h_1}}; per seed {{p5_w20h_2}} / {{p5_w20h_3}} / {{p5_w20h_4}}; verdict {{p5_w20h_5}}
- against the rival at its best over every fit that ran (10× and 30× included): {{p5_w20h_6}}
- the draft's old margin against the 16k-row comparators, recomputed from the same score files: {{p5_w20h_7}} … {{p5_w20h_8}}

| store | seed | strongest rival | store | rival | Δ | CI95 | won/tied/lost | {{@W20M:84}} |
|---|---|---|---|---|---|---|---|---|
| store_K32_n131072 | 0 | mlp__tenx@test | {{p5_w20p3_1:v}} | {{p5_w20p3_2:v}} | **{{p5_w20p3_3:v}}** | [{{p5_w20p3_4:v}}, {{p5_w20p3_5:v}}] | 76/11/22 | {{@W20M:86}} |
| store_K32_n131072 | 1 | mlp__tenx@test | {{p5_w20p3_6:v}} | {{p5_w20p3_7:v}} | **{{p5_w20p3_8:v}}** | [{{p5_w20p3_9:v}}, {{p5_w20p3_10:v}}] | 84/9/16 | {{@W20M:87}} |
| store_K32_n131072 | 2 | mlp__tenx@test | {{p5_w20p3_11:v}} | {{p5_w20p3_12:v}} | **{{p5_w20p3_13:v}}** | [{{p5_w20p3_14:v}}, {{p5_w20p3_15:v}}] | 69/27/13 | {{@W20M:88}} |
| store_K32_n131072 | **mean** | | | | **{{p5_w20p3_16:v}}** | | verdict **SHRINKS** (draft: +39…+49) | {{@W20M:89}} |

### 14.2 P1 convergence: DECLARED against 10× the steps, per family and seed (the carried fit is the last column)

| family | seed | DECLARED | TENX | TENX − DECLARED | P1 | carried | {{@W20M:67}} |
|---|---|---|---|---|---|---|---|
| mlp | 0 | {{p5_w20p1_1:v}} | {{p5_w20p1_2:v}} | {{p5_w20p1_3:v}} | UNDER-TRAINED: TENX carries | tenx | {{@W20M:69}} |
| mlp | 1 | {{p5_w20p1_4:v}} | {{p5_w20p1_5:v}} | {{p5_w20p1_6:v}} | UNDER-TRAINED: TENX carries | tenx | {{@W20M:70}} |
| mlp | 2 | {{p5_w20p1_7:v}} | {{p5_w20p1_8:v}} | {{p5_w20p1_9:v}} | PASS | declared | {{@W20M:71}} |
| lstm | 0 | {{p5_w20p1_10:v}} | {{p5_w20p1_11:v}} | {{p5_w20p1_12:v}} | PASS | declared | {{@W20M:72}} |
| lstm | 1 | {{p5_w20p1_13:v}} | {{p5_w20p1_14:v}} | {{p5_w20p1_15:v}} | PASS | declared | {{@W20M:73}} |
| lstm | 2 | {{p5_w20p1_16:v}} | {{p5_w20p1_17:v}} | {{p5_w20p1_18:v}} | PASS | declared | {{@W20M:74}} |
| transformer | 0 | {{p5_w20p1_19:v}} | {{p5_w20p1_20:v}} | {{p5_w20p1_21:v}} | PASS | declared | {{@W20M:75}} |
| transformer | 1 | {{p5_w20p1_22:v}} | {{p5_w20p1_23:v}} | {{p5_w20p1_24:v}} | PASS | declared | {{@W20M:76}} |
| transformer | 2 | {{p5_w20p1_25:v}} | {{p5_w20p1_26:v}} | {{p5_w20p1_27:v}} | PASS | declared | {{@W20M:77}} |
| cont_mlp | 0 | {{p5_w20p1_28:v}} | {{p5_w20p1_29:v}} | {{p5_w20p1_30:v}} | UNDER-TRAINED: TENX carries | tenx | {{@W20M:78}} |
| cont_mlp | 1 | {{p5_w20p1_31:v}} | {{p5_w20p1_32:v}} | {{p5_w20p1_33:v}} | UNDER-TRAINED: TENX carries | tenx | {{@W20M:79}} |
| cont_mlp | 2 | {{p5_w20p1_34:v}} | {{p5_w20p1_35:v}} | {{p5_w20p1_36:v}} | UNDER-TRAINED: TENX carries | tenx | {{@W20M:80}} |

### 14.3 Table M-131k: the arms (rows 3–4, TEST; fit rows in the arm label)

| arm | role | all corners mean ± sd | per seed | row 3 (pair) | row 4 (n-way) | ceiling (all) | n seeds | {{@W20M:9}} |
|---|---|---|---|---|---|---|---|---|
| store K*=32, fit {{p5_w20a_1:v}} rows | THE ARM | {{p5_w20a_2:v}} ±{{p5_w20a_3:v}} | {{p5_w20a_4:v}} / {{p5_w20a_5:v}} / {{p5_w20a_6:v}} | {{p5_w20a_7:v}} ±{{p5_w20a_8:v}} | {{p5_w20a_9:v}} ±{{p5_w20a_10:v}} | {{p5_w20a_11:v}} | 3 | {{@W20M:11}} |
| store K=5, fit {{p5_w20a_12:v}} rows | THE ARM (K=5) | {{p5_w20a_13:v}} ±{{p5_w20a_14:v}} | {{p5_w20a_15:v}} / {{p5_w20a_16:v}} / {{p5_w20a_17:v}} | {{p5_w20a_18:v}} ±{{p5_w20a_19:v}} | {{p5_w20a_20:v}} ±{{p5_w20a_21:v}} | {{p5_w20a_22:v}} | 3 | {{@W20M:12}} |
| store K*=32, fit {{p5_w20a_23:v}} rows | M-16k arm | {{p5_w20a_24:v}} ±{{p5_w20a_25:v}} | {{p5_w20a_26:v}} / {{p5_w20a_27:v}} / {{p5_w20a_28:v}} | {{p5_w20a_29:v}} ±{{p5_w20a_30:v}} | {{p5_w20a_31:v}} ±{{p5_w20a_32:v}} | {{p5_w20a_33:v}} | 3 | {{@W20M:13}} |
| store K*=32, fit {{p5_w20a_34:v}} rows | M-16k arm | {{p5_w20a_35:v}} ±{{p5_w20a_36:v}} | {{p5_w20a_37:v}} / {{p5_w20a_38:v}} / {{p5_w20a_39:v}} | {{p5_w20a_40:v}} ±{{p5_w20a_41:v}} | {{p5_w20a_42:v}} ±{{p5_w20a_43:v}} | {{p5_w20a_44:v}} | 3 | {{@W20M:15}} |
| prototype table | FLOOR, never a rival | {{p5_w20a_45:v}} ±{{p5_w20a_46:v}} | {{p5_w20a_47:v}} / {{p5_w20a_48:v}} / {{p5_w20a_49:v}} | {{p5_w20a_50:v}} ±{{p5_w20a_51:v}} | {{p5_w20a_52:v}} ±{{p5_w20a_53:v}} | {{p5_w20a_54:v}} | 3 | {{@W20M:17}} |
| mlp, {{p5_w20a_55:v}} rows, DECLARED (spoke-A protocol) | DATA-MATCHED RIVAL | {{p5_w20a_56:v}} ±{{p5_w20a_57:v}} | {{p5_w20a_58:v}} / {{p5_w20a_59:v}} / {{p5_w20a_60:v}} | {{p5_w20a_61:v}} ±{{p5_w20a_62:v}} | {{p5_w20a_63:v}} ±{{p5_w20a_64:v}} | {{p5_w20a_65:v}} | 3 | {{@W20M:21}} |
| mlp, {{p5_w20a_66:v}} rows, 10x steps | 10x DIAGNOSTIC | {{p5_w20a_67:v}} ±{{p5_w20a_68:v}} | {{p5_w20a_69:v}} / {{p5_w20a_70:v}} / {{p5_w20a_71:v}} | {{p5_w20a_72:v}} ±{{p5_w20a_73:v}} | {{p5_w20a_74:v}} ±{{p5_w20a_75:v}} | {{p5_w20a_76:v}} | 3 | {{@W20M:22}} |
| lstm, {{p5_w20a_77:v}} rows, DECLARED (spoke-A protocol) | DATA-MATCHED RIVAL | {{p5_w20a_78:v}} ±{{p5_w20a_79:v}} | {{p5_w20a_80:v}} / {{p5_w20a_81:v}} / {{p5_w20a_82:v}} | {{p5_w20a_83:v}} ±{{p5_w20a_84:v}} | {{p5_w20a_85:v}} ±{{p5_w20a_86:v}} | {{p5_w20a_87:v}} | 3 | {{@W20M:24}} |
| transformer, {{p5_w20a_88:v}} rows, DECLARED (spoke-A protocol) | DATA-MATCHED RIVAL | {{p5_w20a_89:v}} ±{{p5_w20a_90:v}} | {{p5_w20a_91:v}} / {{p5_w20a_92:v}} / {{p5_w20a_93:v}} | {{p5_w20a_94:v}} ±{{p5_w20a_95:v}} | {{p5_w20a_96:v}} ±{{p5_w20a_97:v}} | {{p5_w20a_98:v}} | 3 | {{@W20M:27}} |
| cont_mlp, {{p5_w20a_99:v}} rows, 10x steps | 10x DIAGNOSTIC | {{p5_w20a_100:v}} ±{{p5_w20a_101:v}} | {{p5_w20a_102:v}} / {{p5_w20a_103:v}} / {{p5_w20a_104:v}} | {{p5_w20a_105:v}} ±{{p5_w20a_106:v}} | {{p5_w20a_107:v}} ±{{p5_w20a_108:v}} | {{p5_w20a_109:v}} | 3 | {{@W20M:31}} |
| MLP {{p5_w20a_110:v}} p, {{p5_w20a_111:v}} rows, 20 ep lr1e-3 (existing learned_ref_ownpool) | EXISTING data-matched-at-{{p5_w20a_112:v}} rival — NOT {{p5_w20a_113:v}} | {{p5_w20a_114:v}} ±{{p5_w20a_115:v}} | {{p5_w20a_116:v}} / {{p5_w20a_117:v}} / {{p5_w20a_118:v}} | {{p5_w20a_119:v}} ±{{p5_w20a_120:v}} | {{p5_w20a_121:v}} ±{{p5_w20a_122:v}} | — | 3 | {{@W20M:18}} |
| mlp__declared@test30x | mlp__declared@test30x | {{p5_w20a_123:v}} ±{{p5_w20a_124:v}} | {{p5_w20a_125:v}} / {{p5_w20a_126:v}} / {{p5_w20a_127:v}} | {{p5_w20a_128:v}} ±{{p5_w20a_129:v}} | {{p5_w20a_130:v}} ±{{p5_w20a_131:v}} | {{p5_w20a_132:v}} | 3 | {{@W20M:40}} |

### 14.4 Matched deletion: the four measurements, every scope and seed (rows 3–4 survivor corners, the store's full pool)

| scope | seed | deleted rows | (a) store max|mode dev| | (b1) statistic dev | (b1) statistic bytes | (b2) rendered rel dev | (c) DISPLAY refusal (deleted / false on survivors) | (d) survivors: before → after (Δ) | (d) downdate − refit-without | {{@W20M:210}} |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| value | 0 | 11044 | {{p5_w20d_1:v}} | {{p5_w20d_2:v}} | {{p5_w20d_3:v}} GB | {{p5_w20d_4:v}} | {{p5_w20d_5:v}} / {{p5_w20d_6:v}} | {{p5_w20d_7:v}} → {{p5_w20d_8:v}} ({{p5_w20d_9:v}}) | {{p5_w20d_10:v}} | {{@W20M:212}} |
| value | 1 | 11044 | {{p5_w20d_11:v}} | {{p5_w20d_12:v}} | {{p5_w20d_13:v}} GB | {{p5_w20d_14:v}} | {{p5_w20d_15:v}} / {{p5_w20d_16:v}} | {{p5_w20d_17:v}} → {{p5_w20d_18:v}} ({{p5_w20d_19:v}}) | {{p5_w20d_20:v}} | {{@W20M:213}} |
| value | 2 | 11044 | {{p5_w20d_21:v}} | {{p5_w20d_22:v}} | {{p5_w20d_23:v}} GB | {{p5_w20d_24:v}} | {{p5_w20d_25:v}} / {{p5_w20d_26:v}} | {{p5_w20d_27:v}} → {{p5_w20d_28:v}} ({{p5_w20d_29:v}}) | {{p5_w20d_30:v}} | {{@W20M:214}} |
| conjunction_live | 0 | 5511 | {{p5_w20d_31:v}} | {{p5_w20d_32:v}} | {{p5_w20d_33:v}} GB | {{p5_w20d_34:v}} | — / {{p5_w20d_35:v}} | {{p5_w20d_36:v}} → {{p5_w20d_37:v}} ({{p5_w20d_38:v}}) | {{p5_w20d_39:v}} | {{@W20M:215}} |
| conjunction_live | 1 | 5511 | {{p5_w20d_40:v}} | {{p5_w20d_41:v}} | {{p5_w20d_42:v}} GB | {{p5_w20d_43:v}} | — / {{p5_w20d_44:v}} | {{p5_w20d_45:v}} → {{p5_w20d_46:v}} ({{p5_w20d_47:v}}) | {{p5_w20d_48:v}} | {{@W20M:216}} |
| conjunction_live | 2 | 5511 | {{p5_w20d_49:v}} | {{p5_w20d_50:v}} | {{p5_w20d_51:v}} GB | {{p5_w20d_52:v}} | — / {{p5_w20d_53:v}} | {{p5_w20d_54:v}} → {{p5_w20d_55:v}} ({{p5_w20d_56:v}}) | {{p5_w20d_57:v}} | {{@W20M:217}} |
| item | 0 | 1311 | {{p5_w20d_58:v}} | {{p5_w20d_59:v}} | {{p5_w20d_60:v}} GB | {{p5_w20d_61:v}} | N/A (no hole exists) | {{p5_w20d_62:v}} → {{p5_w20d_63:v}} ({{p5_w20d_64:v}}) | {{p5_w20d_65:v}} | {{@W20M:218}} |
| item | 1 | 1311 | {{p5_w20d_66:v}} | {{p5_w20d_67:v}} | {{p5_w20d_68:v}} GB | {{p5_w20d_69:v}} | N/A (no hole exists) | {{p5_w20d_70:v}} → {{p5_w20d_71:v}} ({{p5_w20d_72:v}}) | {{p5_w20d_73:v}} | {{@W20M:219}} |
| item | 2 | 1311 | {{p5_w20d_74:v}} | {{p5_w20d_75:v}} | {{p5_w20d_76:v}} GB | {{p5_w20d_77:v}} | N/A (no hole exists) | {{p5_w20d_78:v}} → {{p5_w20d_79:v}} ({{p5_w20d_80:v}}) | {{p5_w20d_81:v}} | {{@W20M:220}} |

Statistic size and book refit, from the JSON: the book's exact statistic is {{p5_w20d_82}} B (the fit data itself); (d) on the conjunction at seed 0 is {{p5_w20d_83}}, and downdate − refit-without is {{p5_w20d_84}}.

Rivals' retrain-without on the same rows (the conjunction scope, every family and seed):

| scope | seed | family | rival as-trained → retrain-without (survivors) | ours after | ours − rival retrain-without | CI95 | rival s | rival rows read | rival bytes rewritten | {{@W20M:222}} |
|---|---|---|---|---|---|---|---|---|---|---|
| conjunction_live | 0 | mlp | {{p5_w20r_1:v}} → {{p5_w20r_2:v}} | {{p5_w20r_3:v}} | **{{p5_w20r_4:v}}** | [{{p5_w20r_5:v}}, {{p5_w20r_6:v}}] | {{p5_w20r_7:v}} | 125561 | 522056 | {{@W20M:236}} |
| conjunction_live | 1 | mlp | {{p5_w20r_8:v}} → {{p5_w20r_9:v}} | {{p5_w20r_10:v}} | **{{p5_w20r_11:v}}** | [{{p5_w20r_12:v}}, {{p5_w20r_13:v}}] | {{p5_w20r_14:v}} | 125561 | 522056 | {{@W20M:237}} |
| conjunction_live | 2 | mlp | {{p5_w20r_15:v}} → {{p5_w20r_16:v}} | {{p5_w20r_17:v}} | **{{p5_w20r_18:v}}** | [{{p5_w20r_19:v}}, {{p5_w20r_20:v}}] | {{p5_w20r_21:v}} | 125561 | 522056 | {{@W20M:238}} |
| conjunction_live | 0 | lstm | {{p5_w20r_22:v}} → {{p5_w20r_23:v}} | {{p5_w20r_24:v}} | **{{p5_w20r_25:v}}** | [{{p5_w20r_26:v}}, {{p5_w20r_27:v}}] | {{p5_w20r_28:v}} | 125561 | 522884 | {{@W20M:239}} |
| conjunction_live | 1 | lstm | {{p5_w20r_29:v}} → {{p5_w20r_30:v}} | {{p5_w20r_31:v}} | **{{p5_w20r_32:v}}** | [{{p5_w20r_33:v}}, {{p5_w20r_34:v}}] | {{p5_w20r_35:v}} | 125561 | 522884 | {{@W20M:240}} |
| conjunction_live | 2 | lstm | {{p5_w20r_36:v}} → {{p5_w20r_37:v}} | {{p5_w20r_38:v}} | **{{p5_w20r_39:v}}** | [{{p5_w20r_40:v}}, {{p5_w20r_41:v}}] | {{p5_w20r_42:v}} | 125561 | 522884 | {{@W20M:241}} |
| conjunction_live | 0 | transformer | {{p5_w20r_43:v}} → {{p5_w20r_44:v}} | {{p5_w20r_45:v}} | **{{p5_w20r_46:v}}** | [{{p5_w20r_47:v}}, {{p5_w20r_48:v}}] | {{p5_w20r_49:v}} | 125561 | 521732 | {{@W20M:242}} |
| conjunction_live | 1 | transformer | {{p5_w20r_50:v}} → {{p5_w20r_51:v}} | {{p5_w20r_52:v}} | **{{p5_w20r_53:v}}** | [{{p5_w20r_54:v}}, {{p5_w20r_55:v}}] | {{p5_w20r_56:v}} | 125561 | 521732 | {{@W20M:243}} |
| conjunction_live | 2 | transformer | {{p5_w20r_57:v}} → {{p5_w20r_58:v}} | {{p5_w20r_59:v}} | **{{p5_w20r_60:v}}** | [{{p5_w20r_61:v}}, {{p5_w20r_62:v}}] | {{p5_w20r_63:v}} | 125561 | 521732 | {{@W20M:244}} |
| conjunction_live | 0 | cont_mlp | {{p5_w20r_64:v}} → {{p5_w20r_65:v}} | {{p5_w20r_66:v}} | **{{p5_w20r_67:v}}** | [{{p5_w20r_68:v}}, {{p5_w20r_69:v}}] | {{p5_w20r_70:v}} | 125561 | 527564 | {{@W20M:245}} |
| conjunction_live | 1 | cont_mlp | {{p5_w20r_71:v}} → {{p5_w20r_72:v}} | {{p5_w20r_73:v}} | **{{p5_w20r_74:v}}** | [{{p5_w20r_75:v}}, {{p5_w20r_76:v}}] | {{p5_w20r_77:v}} | 125561 | 527564 | {{@W20M:246}} |
| conjunction_live | 2 | cont_mlp | {{p5_w20r_78:v}} → {{p5_w20r_79:v}} | {{p5_w20r_80:v}} | **{{p5_w20r_81:v}}** | [{{p5_w20r_82:v}}, {{p5_w20r_83:v}}] | {{p5_w20r_84:v}} | 125561 | 527564 | {{@W20M:247}} |

⛔ **A report-vs-artefact disagreement in w20's own `FINDINGS.md`.** It says: {{p5_w20x_1}}. The rows above say otherwise at seed 2: the MLP goes {{p5_w20x_2}} and the continuous MLP {{p5_w20x_3}}. **The artefact wins.** (w22 draft-rulings found the same; the drafts print each rival's before → after.)

### 14.5 The ladder, data-matched set (each rung's own pool): store minus the strongest data-matched rival

- n = 2: store_K32_n131072 − strongest DATA-MATCHED rival ({{p5_w20l_1:v}} rung rows): per seed ['{{p5_w20l_2:v}}', '{{p5_w20l_3:v}}', '{{p5_w20l_4:v}}'], mean **{{p5_w20l_5:v}}** · draft-style margin (− strongest spoke-A `@ref` rival, {{p5_w20l_6:v}} Montero rows): ['{{p5_w20l_7:v}}', '{{p5_w20l_8:v}}', '{{p5_w20l_9:v}}'] {{@W20M:306}}
- n = 3: store_K32_n131072 − strongest DATA-MATCHED rival ({{p5_w20l_10:v}} rung rows): per seed ['{{p5_w20l_11:v}}', '{{p5_w20l_12:v}}', '{{p5_w20l_13:v}}'], mean **{{p5_w20l_14:v}}** · draft-style margin (− strongest spoke-A `@ref` rival, {{p5_w20l_15:v}} Montero rows): ['{{p5_w20l_16:v}}', '{{p5_w20l_17:v}}', '{{p5_w20l_18:v}}'] {{@W20M:342}}
- n = 4: store_K32_n131072 − strongest DATA-MATCHED rival ({{p5_w20l_19:v}} rung rows): per seed ['{{p5_w20l_20:v}}', '{{p5_w20l_21:v}}', '{{p5_w20l_22:v}}'], mean **{{p5_w20l_23:v}}** · draft-style margin (− strongest spoke-A `@ref` rival, {{p5_w20l_24:v}} Montero rows): ['{{p5_w20l_25:v}}', '{{p5_w20l_26:v}}', '{{p5_w20l_27:v}}'] {{@W20M:378}}
- n = 5: store_K32_n131072 − strongest DATA-MATCHED rival ({{p5_w20l_28:v}} rung rows): per seed ['{{p5_w20l_29:v}}', '{{p5_w20l_30:v}}', '{{p5_w20l_31:v}}'], mean **{{p5_w20l_32:v}}** · draft-style margin (− strongest spoke-A `@ref` rival, {{p5_w20l_33:v}} Montero rows): ['{{p5_w20l_34:v}}', '{{p5_w20l_35:v}}', '{{p5_w20l_36:v}}'] {{@W20M:414}}
- n = 6: store_K32_n131072 − strongest DATA-MATCHED rival ({{p5_w20l_37:v}} rung rows): per seed ['{{p5_w20l_38:v}}', '{{p5_w20l_39:v}}', '{{p5_w20l_40:v}}'], mean **{{p5_w20l_41:v}}** · draft-style margin (− strongest spoke-A `@ref` rival, {{p5_w20l_42:v}} Montero rows): ['{{p5_w20l_43:v}}', '{{p5_w20l_44:v}}', '{{p5_w20l_45:v}}'] {{@W20M:452}}

## 15. WAVE 21. FOLDED from each unit's own TABLES.md and run JSONs

### 15.1 `words-full-test`: CMAZE-T, condition WORDS-FULL (the 318-wide design given to EVERY arm), TEST, 3 seeds. **LOSS**

| leg | row | n_seeds | CLU (s0 / s1 / s2) | CLU mean | R* (arm) | R* (s0 / s1 / s2) | R* mean | Δ per seed | Δ̄ | one seed's spread σ | seeds CLU ahead | store gate | LABEL | {{@WFT:8}} |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TEST | `nway` | 3 | {{p5_wf_1:v}} / {{p5_wf_2:v}} / {{p5_wf_3:v}} | {{p5_wf_4:v}} | rival: transformer | {{p5_wf_5:v}} / {{p5_wf_6:v}} / {{p5_wf_7:v}} | {{p5_wf_8:v}} | {{p5_wf_9:v}} / {{p5_wf_10:v}} / {{p5_wf_11:v}} | {{p5_wf_12:v}} | {{p5_wf_13:v}} | 0/3 | holds | **LOSS** | {{@WFT:10}} |
| VAL (w19) | `nway` | 3 | {{p5_wf_14:v}} / {{p5_wf_15:v}} / {{p5_wf_16:v}} | {{p5_wf_17:v}} | rival: transformer | {{p5_wf_18:v}} / {{p5_wf_19:v}} / {{p5_wf_20:v}} | {{p5_wf_21:v}} | {{p5_wf_22:v}} / {{p5_wf_23:v}} / {{p5_wf_24:v}} | {{p5_wf_25:v}} | {{p5_wf_26:v}} | 0/3 | holds | **LOSS** | {{@WFT:11}} |

From the label JSON: {{p5_wf_27}}; mean paired difference {{p5_wf_28}}; seeds with the CLU ahead {{p5_wf_29}} of 3.

### 15.2 `maze-t-seed4`: the SIX-seed table (BOTH, TEST, n-way), the per-seed CEILING row, and the easy/hard split

The cut: {{p5_s4_1}}; easy {{p5_s4_2}}. Six seeds tried, six kept, none discarded.

| seed | origin | BOTH TEST ceiling | reached by | CLU | CLU − ceiling | class | OBS VAL ceiling | WORDS-FULL TEST ceiling | CLU-REL-A10 BOTH VAL (CLU / ceiling) | zero-fit: unidentified n-way rows (TEST) | …also unreachable by colour relation | {{@S4T:15}} |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | original | {{p5_s4c_1:v}} | diagnostic@10x-read-floats: transformer | {{p5_s4c_2:v}} | {{p5_s4c_3:v}} | **easy** | {{p5_s4c_4:v}} | {{p5_s4c_5:v}} | {{p5_s4c_6:v}} / {{p5_s4c_7:v}} | 118 / 761 ({{p5_s4c_8:v}}) | 0 rows | {{@S4T:17}} |
| 1 | original | {{p5_s4c_9:v}} | diagnostic@10x-read-floats: mlp; diagnostic@10x-read-floats: transformer; rival: lstm | {{p5_s4c_10:v}} | {{p5_s4c_11:v}} | **hard** | {{p5_s4c_12:v}} | {{p5_s4c_13:v}} | {{p5_s4c_14:v}} / {{p5_s4c_15:v}} | 403 / 742 ({{p5_s4c_16:v}}) | 79 rows | {{@S4T:18}} |
| 2 | original | {{p5_s4c_17:v}} | CLU | {{p5_s4c_18:v}} | {{p5_s4c_19:v}} | **easy** | {{p5_s4c_20:v}} | {{p5_s4c_21:v}} | {{p5_s4c_22:v}} / {{p5_s4c_23:v}} | 215 / 686 ({{p5_s4c_24:v}}) | 0 rows | {{@S4T:19}} |
| 3 | added | {{p5_s4c_25:v}} | diagnostic@10x-read-floats: transformer | {{p5_s4c_26:v}} | {{p5_s4c_27:v}} | **easy** | {{p5_s4c_28:v}} | — (not run on this seed) | {{p5_s4c_29:v}} / {{p5_s4c_30:v}} | 339 / 675 ({{p5_s4c_31:v}}) | 4 rows | {{@S4T:20}} |
| 4 | added | {{p5_s4c_32:v}} | diagnostic@10x-read-floats: transformer | {{p5_s4c_33:v}} | {{p5_s4c_34:v}} | **hard** | {{p5_s4c_35:v}} | — (not run on this seed) | {{p5_s4c_36:v}} / {{p5_s4c_37:v}} | 305 / 723 ({{p5_s4c_38:v}}) | 0 rows | {{@S4T:21}} |
| 5 | added | {{p5_s4c_39:v}} | diagnostic@10x-read-floats: transformer | {{p5_s4c_40:v}} | {{p5_s4c_41:v}} | **easy** | {{p5_s4c_42:v}} | — (not run on this seed) | {{p5_s4c_43:v}} / {{p5_s4c_44:v}} | 40 / 660 ({{p5_s4c_45:v}}) | 40 rows | {{@S4T:22}} |

| arm | mean all (6) | mean easy [0, 2, 3, 5] | mean hard [1, 4] | mean orig [0, 1, 2] | seed 0 (easy) | seed 1 (hard) | seed 2 (easy) | seed 3 (easy) | seed 4 (hard) | seed 5 (easy) | {{@S4T:40}} |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CLU | {{p5_s4n_1:v}} | {{p5_s4n_2:v}} | {{p5_s4n_3:v}} | {{p5_s4n_4:v}} | {{p5_s4n_5:v}} | {{p5_s4n_6:v}} | {{p5_s4n_7:v}} | {{p5_s4n_8:v}} | {{p5_s4n_9:v}} | {{p5_s4n_10:v}} | {{@S4T:42}} |
| rival (matched, {{p5_s4n_11:v}} steps): lstm | {{p5_s4n_12:v}} | {{p5_s4n_13:v}} | {{p5_s4n_14:v}} | {{p5_s4n_15:v}} | {{p5_s4n_16:v}} | {{p5_s4n_17:v}} | {{p5_s4n_18:v}} | {{p5_s4n_19:v}} | {{p5_s4n_20:v}} | {{p5_s4n_21:v}} | {{@S4T:46}} |
| rival (matched, {{p5_s4n_22:v}} steps): mlp | {{p5_s4n_23:v}} | {{p5_s4n_24:v}} | {{p5_s4n_25:v}} | {{p5_s4n_26:v}} | {{p5_s4n_27:v}} | {{p5_s4n_28:v}} | {{p5_s4n_29:v}} | {{p5_s4n_30:v}} | {{p5_s4n_31:v}} | {{p5_s4n_32:v}} | {{@S4T:47}} |
| rival (matched, {{p5_s4n_33:v}} steps): transformer | {{p5_s4n_34:v}} | {{p5_s4n_35:v}} | {{p5_s4n_36:v}} | {{p5_s4n_37:v}} | {{p5_s4n_38:v}} | {{p5_s4n_39:v}} | {{p5_s4n_40:v}} | {{p5_s4n_41:v}} | {{p5_s4n_42:v}} | {{p5_s4n_43:v}} | {{@S4T:48}} |
| **PER-SEED CEILING** (best trained arm) | {{p5_s4n_44:v}} | {{p5_s4n_45:v}} | {{p5_s4n_46:v}} | {{p5_s4n_47:v}} | **{{p5_s4n_48:v}}** | **{{p5_s4n_49:v}}** | **{{p5_s4n_50:v}}** | **{{p5_s4n_51:v}}** | **{{p5_s4n_52:v}}** | **{{p5_s4n_53:v}}** | {{@S4T:68}} |
| best MATCHED RIVAL per seed (over kinds, input variants, 8k AND 80k) | {{p5_s4n_54:v}} | | | | {{p5_s4n_55:v}} (rival@10x-steps: lstm) | {{p5_s4n_56:v}} (rival: lstm) | {{p5_s4n_57:v}} (rival@10x-steps: lstm) | {{p5_s4n_58:v}} (rival: transformer) | {{p5_s4n_59:v}} (rival@10x-steps: mlp) | {{p5_s4n_60:v}} (rival@10x-steps: mlp) | {{@S4T:71}} |
| CLU − best matched rival (per seed) | {{p5_s4n_61:v}} | | | | {{p5_s4n_62:v}} | {{p5_s4n_63:v}} | {{p5_s4n_64:v}} | {{p5_s4n_65:v}} | {{p5_s4n_66:v}} | {{p5_s4n_67:v}} | {{@S4T:72}} |
| CLU − ceiling | | | | | {{p5_s4n_68:v}} | {{p5_s4n_69:v}} | {{p5_s4n_70:v}} | {{p5_s4n_71:v}} | {{p5_s4n_72:v}} | {{p5_s4n_73:v}} | {{@S4T:73}} |

Per-seed CLU (A1) from the run JSONs: {{p5_s4j_1}} / {{p5_s4j_2}} / {{p5_s4j_3}} / {{p5_s4j_4}} / {{p5_s4j_5}} / {{p5_s4j_6}}

Sensitivity of the split (every possible cut):

| cut between | gap | hard seeds | {{@S4T:28}} |
|---|---|---|---|
| {{p5_s4s_1:v}} – {{p5_s4s_2:v}} | {{p5_s4s_3:v}} | [4] | {{@S4T:30}} |
| {{p5_s4s_4:v}} – {{p5_s4s_5:v}} | {{p5_s4s_6:v}} | [1, 4] | {{@S4T:31}} |
| {{p5_s4s_7:v}} – {{p5_s4s_8:v}} | {{p5_s4s_9:v}} | [1, 3, 4] | {{@S4T:32}} |
| {{p5_s4s_10:v}} – {{p5_s4s_11:v}} | {{p5_s4s_12:v}} | [1, 2, 3, 4] | {{@S4T:33}} |
| {{p5_s4s_13:v}} – {{p5_s4s_14:v}} | {{p5_s4s_15:v}} | [0, 1, 2, 3, 4] | {{@S4T:34}} |

### 15.3 `maze-t-clu-improve` + `rel-newseeds`: CLU-REL-A10 at six seeds, with the `+rel` diagnostic beside it

| quantity | all (6) [0, 1, 2, 3, 4, 5] | orig 0-2 | added 3-5 | easy [0, 2, 3, 5] | hard [1, 4] | {{@RNT:24}} |
|---|---|---|---|---|---|---|
| seed | class (A1 ceiling, maze-t-seed4) | n rows | A1 CLU | A1 ceiling | A1 ceiling arm | A1 best matched rival (arm) | REL CLU | REL ceiling | REL ceiling arm | REL ceiling incl. +rel (arm) | REL best matched rival (arm) | REL best +rel diagnostic (arm) | REL max control (which) | planners ORACLE / BLIND-SEQ / BLIND-B | {{@RNT:13}} |
| A1 best matched rival | {{p5_rn_1:v}} (6) | {{p5_rn_2:v}} (3) | {{p5_rn_3:v}} (3) | {{p5_rn_4:v}} (4) | {{p5_rn_5:v}} (2) | {{@RNT:28}} |
| seed | class (A1 ceiling, maze-t-seed4) | n rows | A1 CLU | A1 ceiling | A1 ceiling arm | A1 best matched rival (arm) | REL CLU | REL ceiling | REL ceiling arm | REL ceiling incl. +rel (arm) | REL best matched rival (arm) | REL best +rel diagnostic (arm) | REL max control (which) | planners ORACLE / BLIND-SEQ / BLIND-B | {{@RNT:13}} |
| seed | class (A1 ceiling, maze-t-seed4) | n rows | A1 CLU | A1 ceiling | A1 ceiling arm | A1 best matched rival (arm) | REL CLU | REL ceiling | REL ceiling arm | REL ceiling incl. +rel (arm) | REL best matched rival (arm) | REL best +rel diagnostic (arm) | REL max control (which) | planners ORACLE / BLIND-SEQ / BLIND-B | {{@RNT:13}} |
| REL best matched rival | {{p5_rn_6:v}} (6) | {{p5_rn_7:v}} (3) | {{p5_rn_8:v}} (3) | {{p5_rn_9:v}} (4) | {{p5_rn_10:v}} (2) | {{@RNT:32}} |
| REL best +rel | {{p5_rn_11:v}} (6) | {{p5_rn_12:v}} (3) | {{p5_rn_13:v}} (3) | {{p5_rn_14:v}} (4) | {{p5_rn_15:v}} (2) | {{@RNT:33}} |
| REL max control | {{p5_rn_16:v}} (6) | {{p5_rn_17:v}} (3) | {{p5_rn_18:v}} (3) | {{p5_rn_19:v}} (4) | {{p5_rn_20:v}} (2) | {{@RNT:34}} |

Per-seed REL CLU from the run JSONs: {{p5_rnj_1}} / {{p5_rnj_2}} / {{p5_rnj_3}} / {{p5_rnj_4}} / {{p5_rnj_5}} / {{p5_rnj_6}}

The `+rel` diagnostic (rivals handed the CLU's relational columns; never a rival row, never a label), n-way, TEST:

| arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | mean (6) | {{@RNT:60}} |
|---|---|---|---|---|---|---|---|---|
| CLU | {{p5_rnr_1:v}} | {{p5_rnr_2:v}} | {{p5_rnr_3:v}} | {{p5_rnr_4:v}} | {{p5_rnr_5:v}} | {{p5_rnr_6:v}} | {{p5_rnr_7:v}} | {{@RNT:62}} |
| rival (matched, {{p5_rnr_8:v}} steps): transformer | {{p5_rnr_9:v}} | {{p5_rnr_10:v}} | {{p5_rnr_11:v}} | {{p5_rnr_12:v}} | {{p5_rnr_13:v}} | {{p5_rnr_14:v}} | {{p5_rnr_15:v}} | {{@RNT:68}} |
| diagnostic+rel@80k: transformer +rel | {{p5_rnr_16:v}} | {{p5_rnr_17:v}} | {{p5_rnr_18:v}} | {{p5_rnr_19:v}} | {{p5_rnr_20:v}} | {{p5_rnr_21:v}} | {{p5_rnr_22:v}} | {{@RNT:71}} |
| **best matched rival (frozen input, 8k/80k)** | {{p5_rnr_23:v}} (rival@10x-steps: mlp; rival@10x-steps: transformer) | {{p5_rnr_24:v}} (rival: transformer; rival@10x-steps: mlp; rival@10x-steps: transformer) | {{p5_rnr_25:v}} (rival: transformer; rival@10x-steps: transformer) | {{p5_rnr_26:v}} (rival: transformer) | {{p5_rnr_27:v}} (rival: transformer) | {{p5_rnr_28:v}} (rival@10x-steps: transformer) | {{p5_rnr_29:v}} | {{@RNT:75}} |
| **best +rel diagnostic (8k/80k)** | {{p5_rnr_30:v}} (diagnostic+rel@80k: mlp +rel) | {{p5_rnr_31:v}} (diagnostic+rel@80k: transformer +rel) | {{p5_rnr_32:v}} (diagnostic+rel@8k: transformer +rel) | {{p5_rnr_33:v}} (diagnostic+rel@8k: transformer +rel) | {{p5_rnr_34:v}} (diagnostic+rel@80k: transformer +rel; diagnostic+rel@8k: transformer +rel) | {{p5_rnr_35:v}} (diagnostic+rel@80k: mlp +rel; diagnostic+rel@80k: transformer +rel; diagnostic+rel@8k: mlp +rel; diagnostic+rel@8k: transformer +rel) | {{p5_rnr_36:v}} | {{@RNT:76}} |
| +rel lift = best +rel − best matched rival | {{p5_rnr_37:v}} | {{p5_rnr_38:v}} | {{p5_rnr_39:v}} | {{p5_rnr_40:v}} | {{p5_rnr_41:v}} | {{p5_rnr_42:v}} | {{p5_rnr_43:v}} | {{@RNT:77}} |
| CLU − best matched rival | {{p5_rnr_44:v}} | {{p5_rnr_45:v}} | {{p5_rnr_46:v}} | {{p5_rnr_47:v}} | {{p5_rnr_48:v}} | {{p5_rnr_49:v}} | {{p5_rnr_50:v}} | {{@RNT:78}} |
| CLU − best +rel | {{p5_rnr_51:v}} | {{p5_rnr_52:v}} | {{p5_rnr_53:v}} | {{p5_rnr_54:v}} | {{p5_rnr_55:v}} | {{p5_rnr_56:v}} | {{p5_rnr_57:v}} | {{@RNT:79}} |
| share of the CLU's margin over the best matched rival that the +rel lift recovers = lift / (CLU − best matched rival) | {{p5_rnr_58:v}} | {{p5_rnr_59:v}} | {{p5_rnr_60:v}} | {{p5_rnr_61:v}} | {{p5_rnr_62:v}} | n/a (margin 0) | — | {{@RNT:80}} |

### 15.4 ⛔⛔ THE TWO TIES, PRINTED AS CELLS. They are measurements, as much as the wins are

**Seed 4: a charter-C control ties the improved CLU.** `control: random_atoms` equals CLU-REL-A10 on n-way:

| seed | leg | config | n | CLU | landscape_removed | landscape_shuffled | landscape_shuffled_across_wells | random_atoms | max control | CLU − max control | control ≥ CLU? | derangement | {{@RNT:165}} |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | TEST | CLU-REL-A10 | 723 | {{p5_tie4_1:v}} | {{p5_tie4_2:v}} | {{p5_tie4_3:v}} | {{p5_tie4_4:v}} | {{p5_tie4_5:v}} | {{p5_tie4_6:v}} (random_atoms) | {{p5_tie4_7:v}} | **TIE/ABOVE: random_atoms** | [1, 0] | {{@RNT:179}} |
| 4 | VAL (free) | CLU-REL-A10 | 696 | {{p5_tie4_8:v}} | {{p5_tie4_9:v}} | {{p5_tie4_10:v}} | {{p5_tie4_11:v}} | {{p5_tie4_12:v}} | {{p5_tie4_13:v}} (random_atoms) | {{p5_tie4_14:v}} | **TIE/ABOVE: random_atoms** | [1, 0] | {{@RNT:180}} |

- run JSON, n-way: {{p5_tie4_15}}
- per held-out tuple (zero-fit replay, every object hash-checked before loading): {{p5_tie4_16}}; objects checked {{p5_tie4_17}}
- the record's own words: {{p5_tie4_18}}

**Seed 5: the best matched rival ties the improved CLU.** The matched transformer at ten times its steps reads what the CLU reads:

- run JSON, n-way: {{p5_tie5_1}}
- the table row: | CLU − best matched rival | {{p5_tie5_2:v}} | {{p5_tie5_3:v}} | {{p5_tie5_4:v}} | {{p5_tie5_5:v}} | {{p5_tie5_6:v}} | {{p5_tie5_7:v}} | {{p5_tie5_8:v}} | {{@RNT:78}} |
- the record's own words: {{p5_tie5_9}}

Where both drafts print them, as measurements: seed 4 in the text, {{p5_dtie_1}} / {{p5_dtie_2}}, and in `tab:cmazet-sym`, {{p5_dtie_3}} / {{p5_dtie_4}}; seed 5 in `tab:cmazet-sym`, {{p5_dtie_5}} / {{p5_dtie_6}}, which equals the store's own seed-5 value in the same row.

So on seed 4 the store is not load-bearing for CLU-REL-A10 (a random-atoms control reads the same on every held-out tuple), and on seed 5 CLU-REL-A10 has no margin over a matched rival. Both drafts print both ties (`tab:cmazet-sym` and the CLU-REL-A10 paragraphs).

### 15.5 `ladder-rivals-32k`: the THIRD rival set (the store-content set). Pool AND steps in every header

The caption the unit ships (every pool and step budget), verbatim: {{p5_lr_1}} rows / {{p5_lr_2}} steps (A); {{p5_lr_3}} rows / {{p5_lr_4}} (B); {{p5_lr_5}} rows / {{p5_lr_6}} plus step-matched (C).

| rung | store K\*=32 (content {{p5_hdr_1:v}} rows) | **A**: K\* − strongest spoke-A `@ref` ({{p5_hdr_2:v}} rows · {{p5_hdr_3:v}} steps) s0/s1/s2 [mean] | **B**: K\* − strongest w20 ({{p5_hdr_4:v}} rows · {{p5_hdr_5:v}} / {{p5_hdr_6:v}} steps) s0/s1/s2 [mean] | **C**: K\* − strongest w21 ({{p5_hdr_7:v}} rows · {{p5_hdr_8:v}} / {{p5_hdr_9:v}} steps) s0/s1/s2 [mean] | **C-sm**: K\* − strongest w21 step-matched to B ({{p5_hdr_10:v}} rows · {{p5_hdr_11:v}} / {{p5_hdr_12:v}} steps) [mean] | `learned_ref_ownpool` ({{p5_hdr_13:v}} rows · {{p5_hdr_14:v}} steps) | n | {{@LRT:11}} |
|---|---|---|---|---|---|---|---|---|
| n=2 | {{p5_lr3_1:v}} | {{p5_lr3_2:v}} / {{p5_lr3_3:v}} / {{p5_lr3_4:v}} [**{{p5_lr3_5:v}}**] | {{p5_lr3_6:v}} / {{p5_lr3_7:v}} / {{p5_lr3_8:v}} [**{{p5_lr3_9:v}}**] | {{p5_lr3_10:v}} / {{p5_lr3_11:v}} / {{p5_lr3_12:v}} [**{{p5_lr3_13:v}}**] | {{p5_lr3_14:v}} / {{p5_lr3_15:v}} / {{p5_lr3_16:v}} [**{{p5_lr3_17:v}}**] | {{p5_lr3_18:v}} | 3 | {{@LRT:13}} |
| n=3 | {{p5_lr3_19:v}} | {{p5_lr3_20:v}} / {{p5_lr3_21:v}} / {{p5_lr3_22:v}} [**{{p5_lr3_23:v}}**] | {{p5_lr3_24:v}} / {{p5_lr3_25:v}} / {{p5_lr3_26:v}} [**{{p5_lr3_27:v}}**] | {{p5_lr3_28:v}} / {{p5_lr3_29:v}} / {{p5_lr3_30:v}} [**{{p5_lr3_31:v}}**] | {{p5_lr3_32:v}} / {{p5_lr3_33:v}} / {{p5_lr3_34:v}} [**{{p5_lr3_35:v}}**] | {{p5_lr3_36:v}} | 3 | {{@LRT:14}} |
| n=4 | {{p5_lr3_37:v}} | {{p5_lr3_38:v}} / {{p5_lr3_39:v}} / {{p5_lr3_40:v}} [**{{p5_lr3_41:v}}**] | {{p5_lr3_42:v}} / {{p5_lr3_43:v}} / {{p5_lr3_44:v}} [**{{p5_lr3_45:v}}**] | {{p5_lr3_46:v}} / {{p5_lr3_47:v}} / {{p5_lr3_48:v}} [**{{p5_lr3_49:v}}**] | {{p5_lr3_50:v}} / {{p5_lr3_51:v}} / {{p5_lr3_52:v}} [**{{p5_lr3_53:v}}**] | {{p5_lr3_54:v}} | 3 | {{@LRT:15}} |
| n=5 | {{p5_lr3_55:v}} | {{p5_lr3_56:v}} / {{p5_lr3_57:v}} / {{p5_lr3_58:v}} [**{{p5_lr3_59:v}}**] | {{p5_lr3_60:v}} / {{p5_lr3_61:v}} / {{p5_lr3_62:v}} [**{{p5_lr3_63:v}}**] | {{p5_lr3_64:v}} / {{p5_lr3_65:v}} / {{p5_lr3_66:v}} [**{{p5_lr3_67:v}}**] | {{p5_lr3_68:v}} / {{p5_lr3_69:v}} / {{p5_lr3_70:v}} [**{{p5_lr3_71:v}}**] | {{p5_lr3_72:v}} | 3 | {{@LRT:16}} |
| n=6 | {{p5_lr3_73:v}} | {{p5_lr3_74:v}} / {{p5_lr3_75:v}} / {{p5_lr3_76:v}} [**{{p5_lr3_77:v}}**] | {{p5_lr3_78:v}} / {{p5_lr3_79:v}} / {{p5_lr3_80:v}} [**{{p5_lr3_81:v}}**] | {{p5_lr3_82:v}} / {{p5_lr3_83:v}} / {{p5_lr3_84:v}} [**{{p5_lr3_85:v}}**] | {{p5_lr3_86:v}} / {{p5_lr3_87:v}} / {{p5_lr3_88:v}} [**{{p5_lr3_89:v}}**] | {{p5_lr3_90:v}} | 3 | {{@LRT:17}} |

Margins, 3-seed mean per rung, from the JSON:

| set | n = 2 | n = 3 | n = 4 | n = 5 | n = 6 | mean over rungs (the unit's FINDINGS; arithmetic of the five JSON leaves) |
|---|---|---|---|---|---|---|
| A (spoke-A `@ref`) | {{p5_lrm_1}} | {{p5_lrm_2}} | {{p5_lrm_3}} | {{p5_lrm_4}} | {{p5_lrm_5}} | {{p5_lrmean_1}} |
| B (w20, data-matched) | {{p5_lrm_6}} | {{p5_lrm_7}} | {{p5_lrm_8}} | {{p5_lrm_9}} | {{p5_lrm_10}} | {{p5_lrmean_2}} |
| C (w21, store-content) | {{p5_lrm_11}} | {{p5_lrm_12}} | {{p5_lrm_13}} | {{p5_lrm_14}} | {{p5_lrm_15}} | {{p5_lrmean_3}} |
| C-sm (w21, step-matched to B) | {{p5_lrm_16}} | {{p5_lrm_17}} | {{p5_lrm_18}} | {{p5_lrm_19}} | {{p5_lrm_20}} | {{p5_lrmean_4}} |

Every bootstrap interval of sets B, C and C-sm lies above zero; the lowest lower bound is {{p5_lr_7}} (n = 2, seed 2, step-matched MLP).

**X1, the single-fit variance (it bears on every single-fit margin in the paper).** Two MLP fits with the same rows, width, schedule and steps, differing only in the random draw:

| rung | ref s0/s1/s2 [mean ± sd] | mine s0/s1/s2 [mean ± sd] | Δ mine − ref | bar (max sd) | per-seed paired Δ [CI95] | verdict | {{@LRT:93}} |
|---|---|---|---|---|---|---|---|
| n=2 | {{p5_x1_1:v}} / {{p5_x1_2:v}} / {{p5_x1_3:v}} [{{p5_x1_4:v}} ± {{p5_x1_5:v}}] | {{p5_x1_6:v}} / {{p5_x1_7:v}} / {{p5_x1_8:v}} [{{p5_x1_9:v}} ± {{p5_x1_10:v}}] | {{p5_x1_11:v}} | {{p5_x1_12:v}} | {{p5_x1_13:v}} [{{p5_x1_14:v}}, {{p5_x1_15:v}}]; {{p5_x1_16:v}} [{{p5_x1_17:v}}, {{p5_x1_18:v}}]; {{p5_x1_19:v}} [{{p5_x1_20:v}}, {{p5_x1_21:v}}] | REPRODUCES within seed spread | {{@LRT:95}} |
| n=3 | {{p5_x1_22:v}} / {{p5_x1_23:v}} / {{p5_x1_24:v}} [{{p5_x1_25:v}} ± {{p5_x1_26:v}}] | {{p5_x1_27:v}} / {{p5_x1_28:v}} / {{p5_x1_29:v}} [{{p5_x1_30:v}} ± {{p5_x1_31:v}}] | {{p5_x1_32:v}} | {{p5_x1_33:v}} | {{p5_x1_34:v}} [{{p5_x1_35:v}}, {{p5_x1_36:v}}]; {{p5_x1_37:v}} [{{p5_x1_38:v}}, {{p5_x1_39:v}}]; {{p5_x1_40:v}} [{{p5_x1_41:v}}, {{p5_x1_42:v}}] | REPRODUCES within seed spread | {{@LRT:96}} |
| n=4 | {{p5_x1_43:v}} / {{p5_x1_44:v}} / {{p5_x1_45:v}} [{{p5_x1_46:v}} ± {{p5_x1_47:v}}] | {{p5_x1_48:v}} / {{p5_x1_49:v}} / {{p5_x1_50:v}} [{{p5_x1_51:v}} ± {{p5_x1_52:v}}] | {{p5_x1_53:v}} | {{p5_x1_54:v}} | {{p5_x1_55:v}} [{{p5_x1_56:v}}, {{p5_x1_57:v}}]; {{p5_x1_58:v}} [{{p5_x1_59:v}}, {{p5_x1_60:v}}]; {{p5_x1_61:v}} [{{p5_x1_62:v}}, {{p5_x1_63:v}}] | REPRODUCES within seed spread | {{@LRT:97}} |
| n=5 | {{p5_x1_64:v}} / {{p5_x1_65:v}} / {{p5_x1_66:v}} [{{p5_x1_67:v}} ± {{p5_x1_68:v}}] | {{p5_x1_69:v}} / {{p5_x1_70:v}} / {{p5_x1_71:v}} [{{p5_x1_72:v}} ± {{p5_x1_73:v}}] | {{p5_x1_74:v}} | {{p5_x1_75:v}} | {{p5_x1_76:v}} [{{p5_x1_77:v}}, {{p5_x1_78:v}}]; {{p5_x1_79:v}} [{{p5_x1_80:v}}, {{p5_x1_81:v}}]; {{p5_x1_82:v}} [{{p5_x1_83:v}}, {{p5_x1_84:v}}] | REPRODUCES within seed spread | {{@LRT:98}} |
| n=6 | {{p5_x1_85:v}} / {{p5_x1_86:v}} / {{p5_x1_87:v}} [{{p5_x1_88:v}} ± {{p5_x1_89:v}}] | {{p5_x1_90:v}} / {{p5_x1_91:v}} / {{p5_x1_92:v}} [{{p5_x1_93:v}} ± {{p5_x1_94:v}}] | {{p5_x1_95:v}} | {{p5_x1_96:v}} | {{p5_x1_97:v}} [{{p5_x1_98:v}}, {{p5_x1_99:v}}]; {{p5_x1_100:v}} [{{p5_x1_101:v}}, {{p5_x1_102:v}}]; {{p5_x1_103:v}} [{{p5_x1_104:v}}, {{p5_x1_105:v}}] | REPRODUCES within seed spread | {{@LRT:99}} |

- the record: {{p5_x1_106}}; {{p5_x1_107}}.

## 16. PLACEHOLDERS AFTER PASS 5 (both drafts, installed bytes)

| placeholder | state | reason |
|---|---|---|
| the ladder's store-content rival set (`tab:ladder`) | ⭐ **FILLED** in both drafts from §15.5 (edit E3). The block prints all eight family fits (selected and ten-times steps), the paired margin row, and the step-matched margin row. Pool and steps are in every block header and in the caption (edits E1, E2) | `w21/ladder-rivals-32k` landed |
| {{p5_ph_1}} (`fig:cmazet-families`) | ⛔ **PERMANENTLY BLOCKED**, left in place | {{p5_ph_2}}. The placeholder index (`app:placeholders`) now says so in both drafts (edit E5) |
| {{p5_ph_3}} (`fig:cmazet-door-tint`) | ⛔ **PERMANENTLY BLOCKED**, left in place | {{p5_ph_4}}. The placeholder index (`app:placeholders`) now says so in both drafts (edit E5) |
| {{p5_ph_5}} (`fig:cmazet-deletion`) | ⛔ **PERMANENTLY BLOCKED**, left in place | {{p5_ph_6}}. The placeholder index (`app:placeholders`) now says so in both drafts (edit E5) |

Counts in the installed drafts: `\PLACEHOLDER{` in the text, 3 each (was 4); `\TODO{` in the text, 4 each (unchanged). The four TODOs are the Head's abstract CMAZE-T sentence, the `pca_tensor` byte ratio, the AI-use statement and the code archive: none is a table cell, and none is this spoke's to fill.

