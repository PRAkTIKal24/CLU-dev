# c3-benchmark-scout — web-scout report

Task + acceptance criterion: Track-A baseline numbers/conventions at 26–47 M on enwik8/WT-103/PG-19 with table-level primary provenance + derived state-bytes; PG-19 GO/CAVEAT/NO-GO with arithmetic; Track-B admissibility scorecard with a runnable criterion-4 tripwire spec and **no** criterion-4 verdict. **Status: done.**

## ⛳ RECONCILIATION LIST (owner needed — protocol §5 corollary, in the first 10 lines)
1. **"Gated DeltaNet-2" is now pinned:** arXiv:**2605.22791** (Hatamizadeh, Choi, Kautz, NVIDIA, 21 May 2026) — *not* Gated DeltaNet v1 (arXiv:2412.06464, ICLR 2025). Any doc citing "GDN-2" must carry the 2605 id.
2. ⛔ **Every modern rival's 26–47 M enwik8 / WT-103 / PG-19 cell is NOT PUBLISHED.** Mamba-2, GDN, GDN-2, TTT and Titans report on The Pile / Books3 / FineWeb-Edu at ≥125 M. Any C2-era doc implying we can *quote* a rival enwik8 number must be corrected: **we train all five arms ourselves** (costed in §1.4 — it is affordable).
3. ⛔ **The dyn-eval anchor 0.94 bpc is at 277 M**, not our weight class. Any table placing 0.94 beside a 40 M number is a category error.
4. ⚠ **"Matched state bytes" is not determined by the architecture.** At a fixed ≈38 M params the natural inference state spans **1.60 MB (TTT-Linear) → 100.7 MB (sliding-window @4 k)**, a **63×** range (§1.5). The budget must be **pre-registered as a number**, not as the word "matched".
5. **PG-19 = GO-WITH-CAVEAT** (§1.6): GO as an internal long-horizon retention venue, **NO-GO as an external comparison venue** — the nearest published numbers are ≥5× our params with a different tokenizer.
6. **A published within-document retention convention EXISTS** and we should adopt rather than invent it: Sun et al. EMNLP 2021's distance-to-last-occurrence bucketing (§1.7). Affects `c3-csf3-harness` §2 (`chlu/eval/text_slices.py`) **while it is being written**.

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)
- **Dial / pillar:** **none — recon/scout.** No performance number of our own, no criterion-4 verdict, no venue adoption. A cited brief + a scorecard; the Hub + Advisor adopt or reject.
- **Laundering control:** n/a directly — but every rival number below is reported **next to the naive/linear/classical number from the same source**, or is explicitly marked as having no such pairing.
- **Falsifies the brief:** a shortlist entry whose criterion-2 pairing I could not find, or whose criterion-4 tripwire I could not spec runnably. (One entry — N-CMAPSS — **does** have a missing criterion-2 pairing; it is declared NOT PUBLISHED and ranked as fallback only, not suppressed.)
- **Does NOT falsify the brief:** zero Track-B survivors. Five of seven screened venues are REJECTs with a named cause of death. ⛔ No survivor was manufactured; the two survivors are `RECOMMEND-IF-TRIPWIRE-CLEARS`, never `RECOMMEND`.

**Pre-registration (§5):** **not applicable** — my acceptance criterion is a cited brief, not a measured ratio/exponent/slope/law. No PREREG.md filed, and no harness of mine measured anything.

**Provenance discipline (hard rule 2):** I ran **zero** experiments. Every number is externally sourced with a table/section citation. Numbers I **derived** (state bytes, PG-19 FLOPs, exemplar counts, dataset-size estimates) are labelled **DERIVED** with the arithmetic shown, and never presented as published. Where a source was unreachable (PDF binary, IdP redirect) I say **NOT OBTAINED** and do not guess.

---

## EXECUTIVE ANSWER (≤10 lines)

1. **Track A, the finding: the 26–47 M enwik8 grid is empty of modern rivals.** The only primary-sourced numbers in our weight class are pre-2023 attention models — **Longformer-small 41 M = 1.00 bpc**, **Adaptive-Span 12L 39 M = 1.02**, **Mega 39 M = 1.02**, **Transformer-XL 12L 41 M = 1.06**. Mamba-2 / GDN / GDN-2 / TTT / Titans publish **nothing** on enwik8, WT-103 or PG-19 at any size.
2. **This is affordable to fix, and that is the actionable conclusion:** a 40 M byte-level arm on enwik8 costs **≈1.5 h (35 % MFU) to ≈18 h (3 % MFU) on 2×A100** — all five rivals × 3 seeds fit the envelope (§1.4).
3. **Dyn-eval substitute column, pinned:** dynamic evaluation of Transformer-XL buys **enwik8 0.99→0.94 bpc** and **WT-103 18.3→16.4 ppl (−9 %)** (Krause et al. 2019). That is the bar our memory dividend is read against — and it is at 277 M, not 41 M.
4. **PG-19: GO-WITH-CAVEAT.** Compute is not the constraint (a full epoch is ≈0.85–9.9 h on 2×A100); comparability is — nearest published models are ≥5× our params.
5. **Track B primary: CAMELS-US rainfall–runoff** — `RECOMMEND-IF-TRIPWIRE-CLEARS`. Passes crit 1/2/3/5 on primary sources; it is the only screened venue where the difficulty is a **published, probe-verified memory of an unobserved accumulated state**.
6. **Track B fallback: N-CMAPSS DS02** — same verdict class, but with a declared **criterion-2 NOT PUBLISHED** and the highest criterion-4 prior on the board.
7. **Five REJECTs with named causes:** LTSF suite (crit 2 — closed-form OLS at the frontier), TSAD suite (crit 2 — a random score is SOTA), traffic (crit 3), PDE rollout (crit 3), streaming-drift (crit 4, already measured and fired).
8. **Biggest single risk:** the matched-state-byte budget (item 4 above) — pick it wrong and the tier-iii control silently decides the result before any physics runs.

---

# 1. TRACK A — the ready spine (bounded-state LM at 26–47 M)

## 1.1 The (rival × venue) grid

Legend: **NP** = NOT PUBLISHED at this weight class · **NC** = published but NOT COMPARABLE · **✓adj** = comparable-adjacent (in weight class, canonical split, with a stated caveat).

| rival | enwik8 (bpc) | WikiText-103 (ppl) | PG-19 (ppl) |
|---|---|---|---|
| **Mamba-2** | **NP** | **NP** | **NP** |
| **Gated DeltaNet-2** (arXiv:2605.22791) | **NP** | **NC** — Wiki ppl 15.90 @1.3 B | **NP** |
| **Gated DeltaNet v1** (arXiv:2412.06464) | **NP** | **NC** — Wiki ppl 16.42 @1.3 B | **NP** |
| **TTT-Linear / TTT-MLP** (arXiv:2407.04620) | **NP** | **NP** | **NP** (Books3 instead) |
| **Titans** (arXiv:2501.00663) | **NP** | **NC** — Wiki ppl 26.18 @340 M | **NP** |
| **Sliding-window attention** (Longformer-small) | **1.00** ✓adj, 41 M | NP | NP |
| **Transformer reference** (Transformer-XL 12L) | **1.06** ✓adj, 41 M | NP at 26–47 M (151 M → 24.0) | NP at 26–47 M (36L → 36.3) |
| **Adaptive-Span 12L** | **1.02** ✓adj, 39 M | — | — |
| **Mega** | **1.02** ✓adj, 39 M | NP at 26–47 M (252 M → 18.07) | — |
| **Compressive Transformer** | 0.97 (24 L, params **not stated**) | — | **33.6** (36 L, params not stated) |
| **dyn-eval column** (Transformer-XL + dyn-eval) | **0.94** ⚠ at 277 M | **16.4** ⚠ at ~257 M | — |

### 1.1.1 Cell detail, with provenance and comparability

**enwik8, in weight class — the four usable anchors.**

- **Longformer, small model, 41 M → test 1.00 bpc** (dev 1.02). Beltagy, Peters, Cohan (2020), *Longformer: The Long-Document Transformer*, arXiv:2004.05150, **Table 2** (small models). *Comparability:* byte-level, vocab 256, canonical enwik8 split ✓; param count in class ✓. **⚠ NOT COMPARABLE on protocol:** staged training over 5 phases from seq-len 2,048 → **23,040**, window sizes 32→512 per layer, dilation on 2 heads in layers 6–11, and **evaluation on sequences of 32,256 tokens**. Quoting 1.00 against a model trained/evaluated at 512–4 k is not like-for-like.
- **Adaptive Attention Span, 12 layers, 39 M → test 1.02 bpc**. Sukhbaatar, Grave, Bojanowski, Joulin (2019), *Adaptive Attention Span in Transformers*, ACL 2019, arXiv:1905.07799, **enwik8 table** (24 L / 209 M → 0.98). ⚠ *Table-number caution:* my retrieval placed enwik8 in Table 2 and text8 in Table 1; I could not re-verify the numbering from a second rendering — **the numbers are confirmed, the table index is single-sourced.** *Comparability:* train block **512 consecutive characters**, attention span limit S=8192; eval context not explicitly stated in the paper.
- **Mega, 39 M → 1.02 bpc**. Ma, Zhou, Kong, He, Gui, Neubig, May, Zettlemoyer (2023), *Mega: Moving Average Equipped Gated Attention*, ICLR 2023, arXiv:2209.10655, **Table 5**. *Comparability:* train chunk 2,048, **eval chunk 4,096**, m∈[2,4] consecutive chunks in training (App. D.3). Length-extrapolating by design.
- **Transformer-XL, 12 layers, 41 M → 1.06 bpc** (18 L/88 M → 1.03; 24 L/277 M → 0.99). Dai, Yang, Yang, Carbonell, Le, Salakhutdinov (2019), *Transformer-XL*, ACL 2019, arXiv:1901.02860, **Table 2**. *Comparability:* **train segment 784, evaluation attention length 3,800** — the bpc is quoted at an eval context ~5× the train context, and the paper says so.

**enwik8, out of class but quoted by the field.** Compressive Transformer 24 L → **0.97 bpc** (Rae, Potapenko, Jayakumar, Lillicrap (2020), *Compressive Transformers for Long-Range Sequence Modelling*, ICLR 2020, arXiv:1911.05507, **Table 4**). ⛔ **Parameter count is not stated in the paper** — do not place it in a param-matched table.

**WikiText-103.** **NOT PUBLISHED at 26–47 M**, word-level or byte-level, in anything I reached. Nearest: Transformer-XL 151 M → **24.0** and 257 M → **18.3** (arXiv:1901.02860, Table 1); Mega 252 M → **18.07** (arXiv:2209.10655, Table 5). **NOT COMPARABLE** to a 26–47 M from-scratch arm.

**The modern rivals — why every cell is NP, stated exactly.**

- **Mamba-2** — Dao & Gu (2024), *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*, ICML 2024, arXiv:2405.21060. Evaluation surface: **MQAR synthetics, Chinchilla scaling laws on The Pile, zero-shot downstream, speed benchmarks.** Verbatim: *"Mamba-2 with 2.7B parameters trained on 300B tokens on the Pile outperforms Mamba-2.8B, Pythia-2.8B and even Pythia-6.9B."* ⛔ **enwik8 and WikiText-103 do not appear.** ⚠ *NOT OBTAINED:* the per-size architecture/recipe appendix table — the arXiv PDF would not parse and the ar5iv rendering returned only front matter. I therefore took Mamba-2's hyperparameters from the **official implementation** instead (§1.5), which is a primary artefact but not the paper's table.
- **Gated DeltaNet-2** — Hatamizadeh, Choi, Kautz (2026), *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention*, arXiv:**2605.22791** (21 May 2026, NVIDIA; code at NVlabs/GatedDeltaNet-2). **Table 2**, recurrent setting, **1.3 B params / 100 B FineWeb-Edu tokens / 4 k seq len**: GDN-2 **Wiki ppl 15.90**, LAMBADA 11.41, avg acc 53.11; KDA 16.81/11.68/52.28; **Gated DeltaNet 16.40**/11.89/52.07; **Mamba-3 (MIMO) 16.45**/11.66/52.39; **Mamba-2 16.79**/12.38/51.82. Hybrid GDN-2: Wiki 15.62 / LMB 10.43. ⛔ **NOT COMPARABLE to WikiText-103 as we would run it**: this is zero-shot perplexity of a 1.3 B model with a subword tokenizer over a fixed 4 k context after 100 B pretraining tokens — a different corpus, tokenizer, param count and protocol on all four axes. It is a *rival-vs-rival* number, not a venue number.
- **Gated DeltaNet v1** — Yang, Kautz, Hatamizadeh (2025), ICLR 2025, arXiv:2412.06464, **Table 3**, 1.3 B / 100 B FineWeb-Edu: GDN **Wiki 16.42 / LMB 12.17**; Mamba2 16.56/12.56; DeltaNet 17.71/16.88; **Transformer++ 18.53/18.32**; Samba 16.13/13.29. ⚠ The 400 M block exists in the same table but its perplexity columns did not separate cleanly in retrieval — **NOT OBTAINED at 400 M**, do not quote.
- **TTT** — Sun, Li, Dalal, Xu, Vikram, Zhang, Dubois, Chen, Wang, Koyejo, Hashimoto, Guestrin (2024), *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*, arXiv:2407.04620. **The Pile** at 2 k/8 k context and **Books3** at 1 k→32 k, at **125 M / 350 M / 760 M / 1.3 B**, Chinchilla recipe; results in **Figures 11–12** (not a single results table). Table 1 is an ablation ladder reaching **ppl 11.09** for TTT-Linear at 125 M. ⛔ No enwik8, no WT-103, no PG-19.
- **Titans** — Behrouz, Zhong, Mirrokni (2025), *Titans: Learning to Memorize at Test Time*, arXiv:2501.00663, **Table 1**: 340 M and 400 M on **15 B FineWeb-Edu tokens**, 760 M on 30 B. Titans (LMM): 340 M **Wiki 26.18 / LMB 29.97**; 400 M **25.03 / 28.99**; 760 M **20.04 / 21.96**. ⛔ No enwik8, no PG-19. ⚠ Also **NOT COMPARABLE**: the 400 M/15 B token budget is ~10× our params and a different corpus.

⚠ **The TTT-class cell matters most (it is our two-sided system-level swap) and it is the emptiest.** There is no published TTT number on any Track-A venue. It must be trained.

## 1.2 The dynamic-evaluation substitute column — convention pinned

**The standing obligation (charter §5) has a settled field convention, and it is Krause's.**

- **Krause, Kahembwe, Murray, Renals (2019), *Dynamic Evaluation of Transformer Language Models*, arXiv:1904.08378.** Abstract, verbatim: *"By applying dynamic evaluation to Transformer-XL models, we improve the state of the art on enwik8 from 0.99 to 0.94 bits/char, text8 from 1.08 to 1.04 bits/char, and WikiText-103 from 18.3 to 16.4 perplexity points."*
- The method itself: **Krause, Kahembwe, Murray, Renals (2018), *Dynamic Evaluation of Neural Sequence Models*, ICML 2018, PMLR 80** — gradient updates on the evaluation stream with a decay toward the initial parameters; the update's λ, η, ε are fit on the validation split.

**What this means for us, stated plainly.** Dyn-eval *is* "the model adapting to recent history" — the exact function a memory claims. Its published purchase is **−0.05 bpc on enwik8 (5.1 % relative)** and **−1.9 ppl / −9 % on WT-103**. Any CLU memory dividend smaller than that, at comparable cost, is not a dividend. **⚠ Both numbers are at 277 M / ~257 M**; the substitute must be **re-measured by us at 26–47 M** because dyn-eval's benefit is known to be scale- and corpus-dependent. Report it as its own column, per arm, never folded into the main number.

## 1.3 Within-document retention / revisit slices — a convention EXISTS; adopt it

`c3-csf3-harness` §2 is building `chlu/eval/text_slices.py` in this same wave. **Do not invent the bucketing.** Three published conventions, in decreasing order of fit:

1. ⭐ **Sun, Krishna, Mattarella-Micke, Iyyer (2021), *Do Long-Range Language Models Actually Use Long-Range Context?*, EMNLP 2021, pp. 807–822, arXiv:2109.09115.** This is the closest thing the field has to *our* instrument, and it is defined **on PG-19**. Their slices:
   - **by distance to last occurrence in the prefix** — tokens whose last occurrence is **>2 K tokens away** (22 K such tokens) vs tokens that **never appear in the prefix** (36 K tokens);
   - **by frequency** — top-10 % most frequent subword types vs the rest (~20 K tokens, ~9 % of the target set);
   - **by subword position** — first subword of a multi-subword word vs the remainder (4.1 K first, 5.1 K rest);
   - **by document type** — fiction vs non-fiction, continuous vs discontinuous narrative.
   Protocol: **220 K validation tokens** sampled from PG-19's 50-book validation set proportional to book length, scored in **target chunks of 10 tokens**, excluding the last 40 tokens of a sequence. Headline finding: long-range context beyond 2 K tokens *"only improves their predictions on a small set of tokens (e.g. those that can be copied from the distant context)."*
2. **Khandelwal, He, Qi, Jurafsky (2018), *Sharp Nearby, Fuzzy Far Away: How Neural Language Models Use Context*, ACL 2018 (aclanthology.org/P18-1027).** The **perturbation-by-distance** protocol: measure the perplexity increase when prior context words are shuffled / replaced / dropped, as a function of distance. Findings: effective context ≈ **200 tokens**; word order matters only within ≈**50 tokens**; beyond that the model retains "a rough semantic field or topic". This is the natural **shuffled-position control** the harness task already specifies — the field has run it, so we can cite the precedent rather than justify it from scratch.
3. **Transformer-XL's RECL** (relative effective context length), Dai et al. 2019 §3.3 — a per-model scalar, coarser than either of the above.

**Recommendation to the Hub (adopt / declare split).**
- **ADOPT** Sun et al.'s *definition* of the retention slice: bucket target positions by **distance to the last occurrence of that same symbol within the same document**, plus the "never seen in prefix" bucket. Cite it; do not rename it.
- **DECLARE AS OURS** (the field has no convention): the **bin edges** at byte level, the fact that we compute it on **enwik8 bytes** rather than PG-19 subwords, computing it **for every arm including the dyn-eval arm**, and the **shuffled-position control** applied to the slice itself. Say so in the paper in one sentence, so a reviewer sees the borrowing and the extension separately.
- ⚠ **Byte-level caveat that will bite:** at vocab 256 the "distance to last occurrence of the same symbol" is a few bytes for common characters and the bucket degenerates. **Define the revisit unit at the word/n-gram level even on a byte stream** (e.g. distance to the previous occurrence of the enclosing whitespace-delimited token), or the slice measures character frequency, not retention. This is the single most likely silent failure in `text_slices.py`.

## 1.4 Reporting norms in this literature (what reviewers will expect)

- **Seeds: one.** Every enwik8/WT-103 table I retrieved (Transformer-XL Table 2, Adaptive-Span, Longformer Table 2, Mega Table 5, Compressive Transformer Tables 3–4) reports **single numbers with no error bars and no seed count**. Our ≥3-seed rule *exceeds* the field norm — say so in the paper; it is free credibility, and it is also why we cannot infer a rival's seed variance from its published number.
- **bpc is quoted at an eval context ≠ train context, and this is disclosed as a feature.** TXL train 784 / eval 3,800; Mega train 2,048 / eval 4,096; Longformer staged to 23,040 / eval 32,256; Adaptive-Span train block 512 / span ≤8,192. **The eval-context ablation is the standard ablation** in this literature and our tables must carry one.
- **Consequence for the matched-state-byte control:** the field's own numbers are obtained at *different* effective state sizes, so "Longformer 1.00 vs TXL-12L 1.06" is already a state-byte-unmatched comparison in the published record. Our matched-bytes table is a genuine methodological contribution here, not a formality.
- **NOT-PUBLISHED cells cost us little to fill. DERIVED arithmetic:** C ≈ 6ND. N = 40 M, D = 5×10⁹ bytes (≈55 epochs of the 90 MB train split, in the range TXL used) ⇒ C = 6 × 4×10⁷ × 5×10⁹ = **1.2×10¹⁸ FLOP**. 2×A100 bf16 peak = 6.24×10¹⁴ FLOP/s.
  | assumed MFU | effective FLOP/s | wall-clock for one arm |
  |---|---|---|
  | 35 % (plain transformer) | 2.18×10¹⁴ | **1.53 h** |
  | 10 % (linear-attn / SSM kernel, conservative) | 6.24×10¹³ | **5.35 h** |
  | 3 % (CHLU block with a multi-step integrator) | 1.87×10¹³ | **17.8 h** |
  **All three fit inside one 2×A100 / 4-day job.** Five rivals × 3 seeds = 15 jobs ⇒ the whole empty grid is fillable. ⚠ This assumes the CHLU block's per-token integrator cost lands ≥3 % MFU; that is the one number the harness must measure early, because at 0.5 % MFU a single arm becomes 4.5 days and breaks the envelope.

## 1.5 State bytes at inference — DERIVED, with the arithmetic

⚠ **DERIVED, not published.** None of the rival papers states an inference state size in bytes at our weight class. I derive from each architecture's own hyperparameters. Assumptions declared: **bf16 = 2 B/element** (fp32 column given too); no quantisation; batch 1; state excludes parameters.

**Our declared reference configs** (chosen to land in 26–47 M at byte-level vocab 256; the harness should pin these or state its own):
- *Attention class:* **12 layers, d_model 512, 8 heads × 64, d_ff 2048** ⇒ params ≈ 12×(4·512² + 2·512·2048) + 256·512 = **37.88 M** ✓ (matches TXL-12L's published 41 M class).
- *Recurrent class* (SSM / linear-attention / TTT, ≈6·d_model² params per layer): **24 layers, d_model 512** ⇒ 24 × 6 × 512² = **37.75 M** ✓.

| arm | state formula | elements | **bytes (bf16)** | bytes (fp32) | ÷ CLU d=12 |
|---|---|---|---|---|---|
| **TTT-Linear**, 24 L, H=8, d_h=64 | n_L·(H·d_h² + H·d_h) | 24×(32,768+512)=798,720 | **1,597,440** | 3,194,880 | **0.81×** |
| **CLU store, d=12** (our reference) | n_atoms×(dim+2)×4 | 491,520 | **1,966,080** | — | 1.00× |
| **Gated DeltaNet / GDN-2**, 24 L, H=4, d_k=d_v=128 | n_L·d_model²/H | 24×65,536=1,572,864 | **3,145,728** | 6,291,456 | 1.60× |
| **Transformer-XL**, 12 L, mem_len 512 | n_L·L·d_model | 12×512×512=3,145,728 | **6,291,456** | 12,582,912 | 3.20× |
| **Mamba-2**, 24 L, d_state 128, headdim 64, expand 2 | n_L·(nh·hd·N + conv_dim·(d_conv−1)) | 24×(131,072+3,840)=3,237,888 | **6,475,776** | 12,951,552 | 3.29× |
| **Sliding-window attn**, 12 L, w=512 | 2·n_L·w·d_model | 2×12×512×512=6,291,456 | **12,582,912** | 25,165,824 | 6.40× |
| **TTT-MLP**, 24 L, H=8, d_h=64 | n_L·H·(8d_h²+5d_h) | 24×264,704=6,352,896 | **12,705,792** | 25,411,584 | 6.46× |
| **Transformer-XL @ eval attn 3800** | n_L·L·d_model | 12×3800×512=23,347,200 | **46,694,400** | 93,388,800 | 23.7× |
| **Sliding-window attn**, 12 L, w=4096 | 2·n_L·w·d_model | 50,331,648 | **100,663,296** | 201,326,592 | 51.2× |

**Derivation provenance for each row**
- *Mamba-2*: official implementation `state-spaces/mamba`, `mamba_ssm/modules/mamba2.py` — defaults `d_state=128, d_conv=4, expand=2, headdim=64, ngroups=1`; `conv_dim = d_ssm + 2*ngroups*d_state`; `allocate_inference_cache` allocates `torch.zeros(batch, nheads, headdim, d_state)`. At d_model 512: d_inner=1024, nheads=16, ssm_state 16·64·128 = 131,072/layer; conv_state (1024+2·1·128)·3 = 3,840/layer. ⚠ Config source is the **code**, not the paper's table (paper appendix NOT OBTAINED).
- *GDN / GDN-2*: from arXiv:2605.22791's own statement for the 1.3 B model — **H=16, d_k=128, d_v=128, d_model=2048, "262,144 floats per batch element" per layer** ⇒ state/layer = d_model²/H exactly. Scaled to d_model 512 at H=4 (keeps d_k=d_v=128 as in the paper). ⚠ **The `flash-linear-attention` library defaults differ** (`hidden_size=2048, head_dim=256, num_heads=6, expand_v=2` ⇒ 6·256·512 = 786,432/layer, **3× the paper's**). ⛔ Pin this in config; do not inherit the library default and then claim byte-matching.
- *TTT*: `test-time-training/ttt-lm-pytorch/ttt.py` — `W1: (num_heads, head_dim, head_dim)`, `b1: (num_heads, 1, head_dim)`; TTT-MLP adds `W2: (num_heads, 4*head_dim, head_dim)`, `b2: (num_heads, 1, head_dim)` (and the symmetric W1 at 4× width). `head_dim = width // num_heads`.
- *Transformer-XL*: caches previous-segment **hidden states** (d_model per position per layer), not separate K/V — hence 1× not 2×. Segment/eval lengths 784 / 3,800 from arXiv:1901.02860.
- *Sliding-window*: caches K **and** V for the last w positions per layer — hence 2×. Longformer's small model uses per-layer windows 32→512 (phase 1) up to 512+ (phase 5), arXiv:2004.05150.
- *CLU d=12*: `.claude/outputs/c2w10-benchmark-gate.md` §5, 1,966,080 B.

⭐ **Two consequences the Hub should act on.**
1. **The natural budgets span 63×** at fixed params (1.60 MB → 100.7 MB). "Matched state bytes" is therefore a **decision**, not a derivation, and whichever number is chosen advantages some rivals and cripples others. **Pre-register the byte budget with a stated rationale before any arm trains.**
2. **The CLU d=12 store (1.97 MB) sits almost exactly on TTT-Linear (1.60 MB, 0.81×)** — which is convenient, because TTT is our two-sided system-level swap. A budget of **≈2 MB** makes the swap byte-honest and simultaneously forces Mamba-2 (3.29×), sliding-window (6.40×) and TXL-at-3800 (23.7×) to be *shrunk to match*, which is the harder and more defensible direction of the control.

## 1.6 PG-19 feasibility at 26–47 M — **GO-WITH-CAVEAT**

**Corpus facts (primary: `google-deepmind/pg19` README + Rae et al. 2020, arXiv:1911.05507 Table 2).**

| split | books | words |
|---|---|---|
| train | **28,602** | **1,973,136,207** |
| validation | **50** | **3,007,061** |
| test | **100** | **6,966,499** |

Full set **28,752 books / 11 GB of text**. Books published pre-1919, Project Gutenberg, minimal preprocessing (boilerplate removal + Ofcom-guided offensive-language mapping).

**Tokenization / protocol convention.** The dataset is **open-vocabulary by design**: the README specifies *"word-level perplexity, by calculating the total likelihood of the dataset (via any chosen subword vocabulary or character-based scheme) divided by the number of tokens."* ⭐ **This is the single best fact in §1.3: a byte-level model can be scored on PG-19 in the venue's own currency, with no tokenizer confound** — the normaliser is the word count, not the token count. Rae et al. themselves used a **32,000-entry SentencePiece/SubwordTextEncoder** vocab and a **512** training window.

**Published numbers (Rae et al. 2020, Table 3):** 36 L Transformer-XL valid **45.5** / test **36.3**; 36 L Compressive Transformer valid **43.4** / test **33.6**. ⛔ **Parameter counts are not stated in the paper.** A 36-layer model at this era's widths is ≥200 M; later PG-19 entries (Routing Transformer, Perceiver AR, Block-Recurrent) are all in the same or larger class. **There is no published PG-19 number within 5× of 26–47 M.**

**Compute, DERIVED, under a 2×A100 / 4-day envelope.** C ≈ 6ND; N = 4.7×10⁷; 2×A100 bf16 peak 6.24×10¹⁴ FLOP/s.
- Chinchilla-optimal D = 20N = 9.4×10⁸ tokens ⇒ C = **2.65×10¹⁷ FLOP**.
- One full epoch: 1.973×10⁹ words × ~1.2 subwords/word (**assumption, declared**) = 2.37×10⁹ tokens ⇒ C = **6.68×10¹⁷ FLOP**.

| MFU | FLOP/s | Chinchilla run | full-epoch run |
|---|---|---|---|
| 35 % | 2.18×10¹⁴ | 0.34 h | **0.85 h** |
| 10 % | 6.24×10¹³ | 1.18 h | **2.97 h** |
| 3 % | 1.87×10¹³ | 3.94 h | **9.92 h** |

**⇒ Compute is not the binding constraint** — even at 3 % MFU a full-epoch PG-19 run is ~10 h against a 96 h limit, and 3 seeds are 3 jobs.

**Verdict: GO-WITH-CAVEAT**, split by purpose:
- ✅ **GO as an internal long-horizon instrument.** PG-19 documents are ~20× longer than WikiText's; it is the *only* Track-A venue where within-document retention over ≥10 K positions is physically present, and **Sun et al.'s published slice convention (§1.3) is defined on exactly this corpus** — so our retention instrument gets a published referent instead of being invented.
- ⛔ **NO-GO as an external comparison venue.** Nearest published numbers are ≥5× our params with a different tokenizer and a 36-layer depth. Any "we get X on PG-19, Rae got 33.6" sentence is **NOT COMPARABLE** and must not be written.
- ⚠ **Engineering caveats that decide whether the seam is cheap:** (i) 11 GB and **~28,752 individual text files** — an inode/many-small-files hazard on CSF3; consolidate once, serially, into a single memmap-able uint8 stream with a sha256 contract, exactly as `chlu/data/enwik8.py` documents; (ii) the **word-count normaliser** must be computed from the raw text, not the tokenizer, or the reported ppl is not the venue's metric; (iii) the validation set is only 50 books / 3.0 M words — small enough that per-book variance is material, so report per-book spread, not just the mean.

This matches the Advisor's ruling ("KEEP THE SEAM"): **GO** ⇒ a small same-wave follow-up cell on the already-landed registry seam.

---

# 2. TRACK B — the Head's direction, screened hard

⭐ **Scorecard artefact: `.claude/outputs/c3-benchmark-scout/trackB-scorecard.json`** (7 ranked rows + a "noted, not recommended" appendix; `crit4_tripwire` carries **no verdict field**, by design).

**Methodology reused, not invented** (task §2.3): `.claude/outputs/c2w10-benchmark-gate.md` (INSECTS — B1 loader positive control → B2 matched-bytes exemplar tripwire → B3 temporal-dependence → B4 byte ledger; anti-hobbling raw+std with the max consumed; the window ladder; No-Change mandatory in every table) and `.claude/outputs/c2w10-metro-gate.md` (Metro — hidden-clock protocol, the **24-h label embargo A(t)**, the relative-margin gate `m2_margin_rel`, the drift map, and the **shuffled-order null**). Both file paths are the ones I read.

## 2.1 Result of the screen

| # | venue | crit 1 | crit 2 | crit 3 | crit 5 | verdict |
|---|---|---|---|---|---|---|
| 1 | **CAMELS-US rainfall–runoff** | PASS | PASS | **PASS** | PASS | **RECOMMEND-IF-TRIPWIRE-CLEARS** (primary) |
| 2 | **N-CMAPSS DS02** | likely | **NOT PUBLISHED** | plausible | partial | **RECOMMEND-IF-TRIPWIRE-CLEARS** (fallback) |
| 3 | METR-LA / PEMS-BAY | PASS | marginal | **FAIL** | partial | REJECT |
| 4 | LTSF suite (ETT/ECL/Traffic/Weather) | pass | **FAIL** | FAIL | FAIL | REJECT |
| 5 | TSAD suite (SMD/SMAP/MSL/SWaT) | **FAIL** | **FAIL** | — | — | REJECT |
| 6 | The Well / PDEBench / PDEArena | strain | pass | **FAIL** | partial | REJECT |
| 7 | INSECTS / Metro / ELEC2 | pass | fail (Metro) | **FAIL** | FAIL | REJECT (crit 4 already fired) |

⛔ **Criterion 4 is not scored anywhere above.** Each row carries a tripwire *spec* in the JSON.

## 2.2 The two known hazards, checked FIRST as instructed

**Hazard A — criterion 2 kills long-horizon forecasting suites. CONFIRMED, and it kills the field's default suite.**
- **Toner & Darlow (2024), *An Analysis of Linear Time Series Forecasting Models*, ICML 2024, PMLR 235, arXiv:2403.14587, Table 2** (ETTm1/m2, ETTh1/h2, ECL, Traffic, Weather, Exchange × horizons 96/192/336/720). They prove DLinear, FITS, RLinear and NLinear are **functionally indistinguishable from unconstrained linear regression** on an augmented feature set, hence admit **closed-form** MSE solutions; and they report the closed-form **OLS outperforms the gradient-trained variants in 23 of 32 settings (72 %)**. Sample cells: ETTh1 T=336, OLS+IN **0.445** vs FITS+IN 0.432; ETTm2 T=720, OLS **0.415** vs FITS 0.409. Headline, verbatim: *"Despite their simplicity, linear models perform well at time series forecasting, even when pitted against deeper and more expensive models."*
- **Zeng, Chen, Zhang, Xu (2023), *Are Transformers Effective for Time Series Forecasting?*, AAAI 2023, arXiv:2205.13504** — LTSF-Linear beats FEDformer by **20–50 %** on multivariate forecasting; >40 % on Exchange, ~30 % on Traffic/Electricity/Weather, ~25 % on ETTm1.
- ⚠ **NOT OBTAINED:** a single table putting closed-form OLS directly beside PatchTST/iTransformer in one source. The Toner & Darlow excerpt I could reach compares OLS against the *linear* family only. **The criterion-2 pairing is nonetheless satisfied** — a one-line closed-form method at the frontier of a suite is disqualifying regardless of which deep model happens to be second.
- **This is Metro Interstate's 2.17 % pathology generalised.** The LTSF suite is therefore **REJECT**, and with it the most obvious reading of "long-horizon multivariate forecasting".

**Hazard B — criterion 4 kills classic streaming venues. ALREADY MEASURED, twice, in this program.** INSECTS `criterion4_cleared = FALSE` at `b2_margin_pts = −1.8983`; Metro `criterion4_cleared_metro = FALSE` at `m2_margin_rel = −0.061539`. Both are closed findings. ⛔ Not re-litigated here.

**Hazard C (new, and I am flagging it because it is the real reason this screen is hard).** The 2026 "long-horizon memory benchmark" literature is **almost entirely agentic/LLM** (LongMemEval-RR, AMA-Bench, DynamicMem, SubtleMemory, EMBER — all 2026 arXiv). There is **no off-the-shelf multivariate-time-series retention benchmark**. Whatever we choose, the retention *instrument* is ours to build — which is consistent with §1.3's finding on the text side.

## 2.3 The primary recommendation — CAMELS-US rainfall–runoff

**Why it survives where the forecasting suites die: the discharge is never an input.** The canonical CAMELS ML task is **simulation, not autoregression** — the model sees 5 daily meteorological forcings + 27 static catchment attributes and must emit discharge. Persistence, seasonal-naive and AR-linear baselines are **structurally unavailable inside the protocol**, so criterion 2 cannot die the way it died on ETT and Metro. The competition is instead a family of decades-tuned, basin-calibrated physical models — which is criterion 1 in its strongest form.

**Criterion 1 + 2, primary source: Kratzert, Klotz, Shalev, Klambauer, Hochreiter, Nearing (2019), *Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets*, HESS 23:5089–5110, Table 3** (447 basins common to all benchmarks; train 1999-10-01→2008-09-30; test 1989-10-01→1999-09-30):

| model | median NSE |
|---|---|
| **EA-LSTM (ensemble)** | **0.74** |
| LSTM with static inputs (ensemble) | 0.72 |
| HBV (upper bound, 100 calibrated) | ~0.67 |
| mHM (basin-calibrated) | ~0.64 |
| VIC (basin-calibrated) | ~0.60 |
| FUSE models | ~0.55–0.60 |
| mHM (regionally calibrated) | 0.53 |
| VIC (regionally calibrated) | 0.31 |
| HBV (lower bound, 1000 uncalibrated) | ~0.20 |
| **per-basin mean flow (the trivial baseline)** | **0.00 by definition** |

⚠ **Single-sourced values flagged:** the four `~` rows are read off Table 3 through one retrieval; **0.74 / 0.72 / 0.53 / 0.31 are the values I am confident in**. Anyone quoting the `~` rows must re-read Table 3.
Headroom: **0.74 − 0.67 = +0.07 median NSE**, i.e. unexplained variance 0.26 vs 0.33 = a **21 % reduction**. Compare the two venues that died: Metro's best strong baseline beat a one-line weekly naive by **2.17 %**.

**Criterion 3 — and this is the reason to prefer it over everything else screened.** The physics of the task *is* memory management: catchment storage (soil moisture, groundwater, **snowpack**) integrates forcings over weeks-to-months and releases them with state-dependent timing. And there is a published, mechanistic demonstration that the winning method's advantage lives in its **memory** specifically: **Lees, Reece, Kratzert, Klotz, Gauch, De Bruijn, Kumar Sahu, Greve, Slater, Dadson (2022), *Hydrological concept formation inside long short-term memory (LSTM) networks*, HESS 26:3079–3101** — linear probes recover **soil-moisture and snow-water-equivalent stores from the LSTM cell state**, despite the network never seeing those variables in training. ⭐ That is criterion 3 satisfied by external evidence rather than by our assertion, and it is directly on the CLU's thesis (a latent state that carries a *physically consistent* store).

**Criterion 5 — all levers live.** Learned **φ** must earn its decomposition from raw meteorology (this is exactly the C2 Add.16 blocker, tested where φ *cannot* be handed its features). Learned **ψ** is a real read-out head. **Lifetimes** have a physical referent (storage residence times) — the first venue where "lifetime is a dial you set" has an external check. The **controller** faces genuine seasonal capacity pressure (snow accumulation = monotonic load, melt = scheduled release). **Retry / anytime reads** have an operational reading at flood peaks.

**Criterion 4: SPEC ONLY — see the JSON row.** In one paragraph: query = causally-standardised 365-day trailing forcing window ⊕ 27 static attributes (1,852 dims; plus a 30-day/177-dim variant); keys = identical windows from the training period, pooled regionally (primary) and same-basin (secondary); exemplar baseline = distance-weighted k-NN, k∈{1,3,5,10,25}, L-ladder {250…at-budget}, **raw and standardised with the max consumed** (the anti-hobbling rule that decided the INSECTS gate); byte budget 1,966,080 B ⇒ **265 exemplars** at 365-day resolution or **2,761** at 30-day; **dies if** the best exemplar arm's median NSE comes within **0.02 absolute (or 2 % relative)** of the best strong reference, reported against **every** strong reference (the INSECTS gate survived scrutiny only because all five ARF references were printed); plus the **Metro shuffled-order null** as the "is there any retention signal at all" leg; plus a **loader positive control first** — CAMELS ships the calibrated benchmark models' own daily output, so we can recompute a published median NSE and require it to land in tolerance before any tripwire number is quoted. Cost: **CPU-only, hours**, comparable to the ~4 h INSECTS and ~1 h Metro precedents.

⭐ **The one honest reason this venue might be the first to survive the criterion-4 theorem** (stated as a hypothesis, not a conclusion): the target is a function of an **unobserved accumulated state**, not of the forcing window. Two identical 365-day forcing windows in basins with different antecedent storage produce different discharge, and the static attributes only partially disambiguate. If that non-identifiability is material, **no metric over the observable window can be the provable ceiling** — which is precisely the structural condition the six previous confirmations all lacked. ⛔ This is a reason to *run the tripwire*, not a reason to skip it.

## 2.4 The fallback — N-CMAPSS DS02, with two declared weaknesses

Head-named prognostics leg; real, public, and long-horizon (whole flight cycles across hundreds of cycles). **Arias Chao, Kulkarni, Goebel, Fink (2021), *Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics*, Data 6(1):5** is the dataset of record; DS02 = 20 channels, 9 units, train {2,5,10,16,18,20} / test {11,14,15}.

Two weaknesses, both declared rather than papered over:
1. ⛔ **Criterion 2 is NOT PUBLISHED.** I found **no published trivial baseline** at this venue. Per hard rule 3 that is a NOT-RUN, never a null and never a win. What it costs *us*: **minutes of CPU** — mean-RUL, affine-in-cycle-index, and per-unit health-index extrapolation. ⚠ The affine-in-cycle-index baseline is historically dangerous on the CMAPSS family because RUL is *defined* as a piecewise-linear function of cycle count.
2. ⚠ **Highest criterion-4 prior on the board.** **Similarity-based RUL estimation** — match the query trajectory against a library of stored run-to-failure trajectories — is a classical, published, competitive method on this family. The venue arrives with a pre-existing reason to be metric-native. Spec'd anyway (JSON row 2), ranked **below** CAMELS for exactly this reason, per hard rule 1.

⛔ **CAFE EMBARGO, in-line as required:** **no CAFE C-MAPSS number is externally comparable** (the banked label-bug report). Nothing from our CAFE prognostics work may be quoted for or against this venue in either direction. N-CMAPSS is a different, externally published dataset — but any of our numbers flowing through CAFE-derived preprocessing inherits the embargo.

Also relayed and **not adopted**: a DS02 RMSE of 5.04 attributed to a deep model in secondary sources. ⛔ I could not reach the originating table. **NOT VERIFIED — do not quote.**

## 2.5 The five REJECTs, with causes of death

- **Traffic (METR-LA / PEMS-BAY)** — **crit 3.** Criteria 1 and 2 actually pass: Wu, Pan, Long, Jiang, Zhang (2019), *Graph WaveNet*, IJCAI 2019, arXiv:1906.00121, **Table 2** — METR-LA 60 min, **Historical Average MAE 4.16** vs **Graph WaveNet 3.53** (15.1 % relative; DCRNN 3.60, STGCN 4.59, FC-LSTM 4.37, ARIMA 6.90); PEMS-BAY 60 min, **HA 2.88** vs **GWNet 1.95** (32.3 %). But the task is **12 steps in → 12 steps out**, i.e. a one-hour horizon from a one-hour window: single-shot context, not memory management, and the published difficulty is **spatial** structure. ⚠ Secondary hazard: HA *is* a time-of-day exemplar store sitting 15 % behind SOTA.
- **LTSF suite** — **crit 2** (§2.2 Hazard A).
- **TSAD suite (SMD/SMAP/MSL/SWaT/PSM)** — **crit 2, at the protocol level.** Kim, Choi, Choi, Lee, Yoon (2022), *Towards a Rigorous Evaluation of Time-series Anomaly Detection*, AAAI 2022, arXiv:2109.05257: the point-adjust protocol *"has a great possibility of overestimating detection performance, with even a random anomaly score easily turning into a state-of-the-art TAD method"*, and an **untrained** model matches existing methods even when PA is forbidden. ⚠ Also a licensing blocker: **SWaT/WADI require an iTrust request form** — Head clearance, not an engineering fix.
- **PDE rollout (The Well / PDEBench / PDEArena)** — **crit 3, plus crit-1 strain.** Ohana et al. (2024), *The Well*, NeurIPS 2024 D&B, arXiv:2412.00568, **Tables 2–3**: VRMSE is normalised so that **predicting the field mean = 1.0**, and on several datasets **all four baseline classes score >1** (on `rayleigh_taylor_instability`, **>>10**). Headroom is enormous but the competition frequently *loses to the field mean*, which is criterion 1 failing from the other side; and the dynamics are fully-observed and near-Markovian, so a memory primitive has no distinctive job. Data facts if ever revisited: **16 datasets, 15 TB total, 6.9 GB–5.1 TB each, CC BY 4.0**, HuggingFace `polymathic-ai/*` + `the-well-download` CLI, streamable.
- **Streaming drift (INSECTS / Metro / ELEC2)** — **crit 4, measured and fired** (§2.2 Hazard B), plus crit 2 on Metro and crit 3 on both (INSECTS exemplar accuracy is *monotonically decreasing* in store size above L≈500; Metro's shuffled-order null showed ordering carries no exploitable information at these budgets). Closed.

**Appendix — noted, not recommended:** WeatherBench 2 / ERA5 (crit 1 and 2 excellent, GraphCast is 36.7 M params = literally our weight class, but crit 3 fails on near-Markovian dynamics and training cost is far outside 2×A100/4-day); agentic LLM-memory benchmarks (⛔ outside the Head's ratified direction; crit 3 satisfied *by construction*, which is why they are not a fair test); battery degradation, Severson et al. 2019 (crit-2 risk decisive — an elastic net on early-cycle features is the published frontier).

---

# 3. CSF3 data availability & licensing

| dataset | source | licence | size | credentials? | CSF3 staging |
|---|---|---|---|---|---|
| **enwik8** | mattmahoney.net/dc/enwik8.zip (DeepAI mirror) | Wikipedia text: CC BY-SA 3.0 / GFDL; Hutter-Prize distribution free | 36,445,475 B zip → **100,000,000 B** payload | **no** | ✅ **built** (`chlu/data/enwik8.py`): download-once, atomic-rename, payload-length guard |
| **WikiText-103-raw** | s3 research.metamind.io (HF Salesforce mirror) | **CC BY-SA 3.0** | ~181 MB zip | **no** | ✅ **built** (`chlu/data/wikitext.py`), byte + word modes |
| **PG-19** | `gs://deepmind-gutenberg`; HF `deepmind/pg19`; TFDS `pg19` | **Apache-2.0** (repo README); books public domain | **11 GB**, ~28,752 files | **no** | ⚠ **not built** — consolidate ~28.7 k files into one memmap stream once, serially, then sha256-freeze. Prefer the HF mirror over gsutil. |
| **CAMELS-US** | NCAR/UCAR direct HTTP | ⚠ **explicit licence string NOT CONFIRMED** — Addor et al. 2017 HESS 21:5293 distributes openly; USGS discharge is US-Gov public domain | ⚠ "15 GB compressed / 130 GB uncompressed" is **SECONDARY-SOURCED ONLY (hyper.ai) — do not quote.** **DERIVED**: forcing+discharge time series alone ≈ 671 basins × ~12,784 days × 6 float32 ≈ **206 MB**; ≈0.6 GB with all three forcing products | **no** | ✅ fits `download_file` pattern; consolidate ascii → one array at stage time. **⛔ Verify the licence string before mirroring — the one open blocker on the primary recommendation, and a 10-minute check.** |
| **Caravan** | Zenodo + HuggingFace | reported **CC-BY-4.0** — ⚠ **SINGLE-SOURCED**; the Scientific Data page (Kratzert et al. 2023, *Sci Data* 10:61) sat behind an IdP redirect and was **NOT OBTAINED** | 6,830 basins core (22,732 with extensions), 1981–2020 daily, netCDF; **size NOT VERIFIED** | **no** | ✅ selected precisely because its constituent licences permit redistribution of daily flow — the reason to prefer it if we ever mirror |
| **N-CMAPSS** | NASA PCoE prognostics repository | US-Gov work; **NOT INDEPENDENTLY VERIFIED** | "several GB" — **NOT VERIFIED**, measure at stage time | reported none | ✅ HDF5; stage once, sha256-freeze |
| **METR-LA / PEMS-BAY** | DCRNN release mirrors | open | <1 GB | no | ✅ trivial |
| **The Well** | HF `polymathic-ai/*`, `the-well-download` | **CC BY 4.0** | 15 TB total; **6.9 GB–5.1 TB** per dataset | no | ⚠ pick ONE 6.9 GB dataset; ⛔ never the full collection |
| **SWaT / WADI** | iTrust, SUTD | **request form + approval** | — | ⛔ **YES** | ⛔ **BLOCKER for the Head, not an engineer** |
| **INSECTS / Metro** | USP DS Repository / UCI | CC BY 4.0 / UCI | frozen | no | ✅ already frozen with sha256 at `.claude/data/c2w10-streams/`, `.claude/data/c2w10-metro/` |

⛔ **Only one credentials blocker in the whole sweep: SWaT/WADI** — and that venue is REJECTed on criterion 2 anyway, so nothing the Head must clear is on the critical path.

---

# 4. Declared NOT-OBTAINED / NOT-VERIFIED (never to be reported as facts)

1. **Mamba-2's per-size architecture & training-recipe appendix table** — arXiv PDF unparseable, ar5iv returned front matter only. Hyperparameters taken from the **official code** instead, and labelled as such.
2. **Mamba-2's Pile perplexity at 130 M / 370 M** — not reached. ⛔ Do not quote.
3. **Gated DeltaNet v1's 400 M perplexity columns** — present in Table 3 but did not separate in retrieval.
4. **Adaptive-Span's table *index*** for enwik8 — the numbers (39 M → 1.02; 209 M → 0.98) are confirmed; the table number is single-sourced.
5. **Compressive Transformer parameter counts** (36 L PG-19, 24 L enwik8) — not stated in the paper.
6. **Toner & Darlow vs PatchTST/iTransformer in one table** — the excerpt reached compares OLS to the linear family only.
7. **N-CMAPSS DS02 baseline RMSE 5.04** — relayed by secondary sources; originating table not reached. ⛔ Do not quote.
8. **CAMELS "15 GB / 130 GB" and Caravan's licence + size** — secondary/blocked sources; flagged in §3.
9. **Kratzert 2019 Table 3's `~` rows** (HBV/mHM/VIC/FUSE) — single retrieval; re-read before quoting.
10. **TTT / Titans / GDN-2 numbers on any Track-A venue** — they do not exist; that is the finding, not a gap in my search.

---

# 5. Git footprint

**None.** Zero worktrees, zero branches, zero commits, zero pushes, zero tracked-file edits. Read-only on the repo; all artefacts under `.claude/`. Files read: `.claude/AGENT_PROTOCOL.md`, `.claude/advisor-head-c3-charter.md`, `.claude/advisor-head-intervention.md`, `.claude/c3-handover.md`, `.claude/tasks/c3-benchmark-scout.md`, `.claude/outputs/c2w10-benchmark-gate.md`, `.claude/outputs/c2w10-metro-gate.md`, `chlu/data/enwik8.py`, `chlu/data/wikitext.py`.
Files written: `.claude/outputs/c3-benchmark-scout.md`, `.claude/outputs/c3-benchmark-scout/trackB-scorecard.json`.

# 6. Open questions / risks for the Hub

1. **The byte budget is the highest-leverage undecided number in the wave** (§1.5). Recommend **≈2 MB**, pre-registered, because it makes the TTT-class swap byte-honest and forces every other rival to be shrunk rather than grown.
2. **`text_slices.py` is being written right now** and §1.3's byte-level caveat (revisit distance must be defined at word/n-gram granularity, not per byte) needs to reach that engineer **this wave**, or the instrument will silently measure character frequency.
3. **Fifteen rival-arm jobs** (5 rivals × 3 seeds) are affordable but need scheduling; §1.4's MFU assumption is the thing that could break the envelope, and the harness should measure the CHLU block's MFU on day one.
4. **CAMELS licence-string verification** is the only Head-facing item on the primary recommendation's critical path.
5. **If the CAMELS tripwire fires**, that is the *seventh* confirmation and the Head/Advisor — not a spoke — must decide whether Track B is a venue at all. ⛔ I did not improvise an eighth candidate, per stop condition §4.

---

## Proposed handover updates (for the Hub)

**State snapshot / C3W1 log — new entries**
- **`c3-benchmark-scout` DONE (2026-08-12, zero worktrees, zero commits).** Track A: the **26–47 M enwik8 grid is empty of modern rivals** — Mamba-2 / GDN / GDN-2 / TTT / Titans publish nothing on enwik8, WT-103 or PG-19 at any size; the only in-class primary-sourced anchors are **Longformer-small 41 M = 1.00 bpc** (arXiv:2004.05150 Table 2), **Adaptive-Span 12L 39 M = 1.02** (arXiv:1905.07799), **Mega 39 M = 1.02** (arXiv:2209.10655 Table 5), **Transformer-XL 12L 41 M = 1.06** (arXiv:1901.02860 Table 2). ⇒ **all five rival arms must be trained by us**, costed at **1.5–17.8 h per arm on 2×A100** (6ND, 40 M × 5e9 bytes, 35 %/10 %/3 % MFU).
- **"Gated DeltaNet-2" pinned to arXiv:2605.22791** (Hatamizadeh, Choi, Kautz, NVIDIA, 2026-05-21). Its 1.3 B/100 B-FineWeb-Edu Table 2: GDN-2 Wiki 15.90 / LMB 11.41; GDN 16.40; Mamba-3 (MIMO) 16.45; Mamba-2 16.79; KDA 16.81. **NOT COMPARABLE** to a 26–47 M WT-103 arm on four axes.
- **Dyn-eval substitute column convention pinned:** Krause et al. 2019 (arXiv:1904.08378) — **enwik8 0.99→0.94 bpc, WT-103 18.3→16.4 ppl (−9 %)**, method = Krause et al. ICML 2018. ⚠ **Both at 277 M / ~257 M** — must be re-measured at our weight class; ⛔ never printed beside a 40 M number.
- **Retention/revisit convention EXISTS — adopt, don't invent:** Sun et al., EMNLP 2021, arXiv:2109.09115 (distance-to-last-occurrence buckets: >2 K away / never-in-prefix; frequency; first-vs-rest subword; 220 K PG-19 validation tokens, 10-token target chunks). Shuffled-position control has a precedent in Khandelwal et al., ACL 2018 (effective context ≈200 tokens; order matters only within ≈50). **Ours to declare:** byte-level bin edges, computing the slice for every arm incl. dyn-eval, and the ⚠ **byte-level caveat** — define the revisit unit at word/n-gram granularity or the slice measures character frequency.
- **⭐ Matched-state-bytes is a DECISION, not a derivation.** At ≈38 M params, DERIVED inference state (bf16): **TTT-Linear 1,597,440 B · CLU d=12 1,966,080 B · GDN 3,145,728 B · TXL(mem 512) 6,291,456 B · Mamba-2 6,475,776 B · sliding-window(512) 12,582,912 B · TTT-MLP 12,705,792 B · TXL(eval 3800) 46,694,400 B · sliding-window(4096) 100,663,296 B** — a **63× span**. Recommend pre-registering **≈2 MB** (the TTT-class swap is then byte-honest and every other rival must be shrunk, not grown). ⚠ `flash-linear-attention`'s GDN defaults give **3×** the paper's per-layer state — pin it in config.
- **PG-19 = GO-WITH-CAVEAT.** Corpus: 28,602 train books / 1,973,136,207 words; 50 / 3,007,061 valid; 100 / 6,966,499 test; 11 GB; **Apache-2.0**; open-vocabulary word-level ppl (normaliser = word count ⇒ **a byte-level model can be scored in the venue's own currency**). Compute is not binding: a **full epoch is 0.85 h (35 % MFU) / 2.97 h (10 %) / 9.92 h (3 %)** on 2×A100. ⛔ **NO-GO as an external comparison venue** — nearest published (Rae et al. ICLR 2020 Table 3: 36 L TXL 36.3, 36 L Compressive 33.6) has **no stated parameter count** and is ≥5× our class.
- **Track B: primary = CAMELS-US rainfall–runoff, `RECOMMEND-IF-TRIPWIRE-CLEARS`; fallback = N-CMAPSS DS02, same class with a declared crit-2 NOT-PUBLISHED.** Five REJECTs with causes: LTSF (crit 2 — closed-form OLS at the frontier, Toner & Darlow ICML 2024), TSAD (crit 2 — random score = SOTA, Kim et al. AAAI 2022), traffic (crit 3), PDE rollout (crit 3 + crit-1 strain, The Well NeurIPS 2024 D&B), streaming drift (crit 4 already fired). Scorecard: `.claude/outputs/c3-benchmark-scout/trackB-scorecard.json`.
- **CAMELS crit-1/2 anchor (Kratzert et al. 2019, HESS 23:5089, Table 3, 447 basins):** EA-LSTM ensemble **median NSE 0.74**, LSTM+static 0.72, HBV-upper ~0.67, mHM-basin ~0.64, VIC-basin ~0.60, mHM-regional 0.53, VIC-regional 0.31, **mean-flow 0.00 by definition**. Crit-3 external evidence: Lees et al. 2022, HESS 26:3079 — linear probes recover **soil moisture and snow-water-equivalent from the LSTM cell state**.

**Registry entries**
- `negative_results.md`: **"Modern SSM/linear-attention rivals have no published number on any Track-A venue at any size."** Mamba-2 → Pile/MQAR; GDN & GDN-2 → FineWeb-Edu Wiki/LMB ppl at 1.3 B; TTT → Pile/Books3 Figs 11–12; Titans → FineWeb-Edu Table 1. Consequence: **every rival cell in the tier-iii table is a from-scratch run of ours**, and there is no external number to check ourselves against except the four pre-2023 attention anchors.
- `negative_results.md`: **"The standard LTSF suite fails criterion 2 as a family"** — closed-form OLS beats gradient-trained linear variants in 23/32 settings (Toner & Darlow, ICML 2024, arXiv:2403.14587, Table 2); LTSF-Linear beats FEDformer by 20–50 % (Zeng et al., AAAI 2023). Metro's 2.17 % pathology generalised.
- `negative_results.md`: **"The 2026 long-horizon-memory benchmark literature is agentic/LLM, not time-series."** No off-the-shelf multivariate-TS retention venue exists ⇒ the retention instrument is ours on **both** tracks.
- `claims_matrix.md`: dyn-eval substitute column now has a **pinned numeric bar** (−0.05 bpc enwik8 / −9 % WT-103, at 277 M, to be re-measured in class).

**Owner needed (§5 corollary)** — the six-item reconciliation list at the top of this report, in particular items 2 (rival cells are NOT-PUBLISHED, not quotable), 4 (pre-register the byte budget) and 6 (the retention-slice convention must reach `c3-csf3-harness` **while it is being written**).

---

## Bibtex-ready references

```bibtex
@inproceedings{dai2019transformerxl,
  title     = {Transformer-{XL}: Attentive Language Models beyond a Fixed-Length Context},
  author    = {Dai, Zihang and Yang, Zhilin and Yang, Yiming and Carbonell, Jaime and Le, Quoc V. and Salakhutdinov, Ruslan},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2019},
  eprint    = {1901.02860},
  archivePrefix = {arXiv},
  note      = {enwik8 Table 2: 12L/41M = 1.06 bpc; 24L/277M = 0.99. WT-103 Table 1: 151M = 24.0}
}
@inproceedings{sukhbaatar2019adaptivespan,
  title     = {Adaptive Attention Span in Transformers},
  author    = {Sukhbaatar, Sainbayar and Grave, Edouard and Bojanowski, Piotr and Joulin, Armand},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2019},
  eprint    = {1905.07799},
  archivePrefix = {arXiv},
  note      = {enwik8: 12L/39M = 1.02 bpc; 24L/209M = 0.98. Train block 512 chars, span limit S=8192}
}
@inproceedings{beltagy2020longformer,
  title  = {Longformer: The Long-Document Transformer},
  author = {Beltagy, Iz and Peters, Matthew E. and Cohan, Arman},
  year   = {2020},
  eprint = {2004.05150},
  archivePrefix = {arXiv},
  note   = {Table 2, small model 41M: enwik8 test 1.00 bpc (dev 1.02), text8 test 1.10. Staged training to seq-len 23040, eval 32256}
}
@inproceedings{ma2023mega,
  title     = {Mega: Moving Average Equipped Gated Attention},
  author    = {Ma, Xuezhe and Zhou, Chunting and Kong, Xiang and He, Junxian and Gui, Liangke and Neubig, Graham and May, Jonathan and Zettlemoyer, Luke},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023},
  eprint    = {2209.10655},
  archivePrefix = {arXiv},
  note      = {Table 5: enwik8 39M = 1.02 bpc (train chunk 2048, eval 4096); WT-103 252M = 18.07}
}
@inproceedings{rae2020compressive,
  title     = {Compressive Transformers for Long-Range Sequence Modelling},
  author    = {Rae, Jack W. and Potapenko, Anna and Jayakumar, Siddhant M. and Lillicrap, Timothy P.},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2020},
  eprint    = {1911.05507},
  archivePrefix = {arXiv},
  note      = {PG-19 Table 3: 36L TXL 45.5/36.3, 36L Compressive 43.4/33.6 (param counts NOT stated). enwik8 Table 4: 24L = 0.97. PG-19 stats Table 2}
}
@article{krause2019dyneval,
  title   = {Dynamic Evaluation of Transformer Language Models},
  author  = {Krause, Ben and Kahembwe, Emmanuel and Murray, Iain and Renals, Steve},
  journal = {arXiv preprint arXiv:1904.08378},
  year    = {2019},
  note    = {enwik8 0.99 -> 0.94 bpc; text8 1.08 -> 1.04; WikiText-103 18.3 -> 16.4 ppl}
}
@inproceedings{krause2018dyneval,
  title     = {Dynamic Evaluation of Neural Sequence Models},
  author    = {Krause, Ben and Kahembwe, Emmanuel and Murray, Iain and Renals, Steve},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning (ICML)},
  series    = {PMLR}, volume = {80}, year = {2018}
}
@inproceedings{dao2024mamba2,
  title     = {Transformers are {SSM}s: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
  author    = {Dao, Tri and Gu, Albert},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024},
  eprint    = {2405.21060},
  archivePrefix = {arXiv},
  note      = {No enwik8 / WikiText-103. Pile scaling laws, MQAR, zero-shot downstream}
}
@inproceedings{yang2025gateddeltanet,
  title     = {Gated Delta Networks: Improving Mamba2 with Delta Rule},
  author    = {Yang, Songlin and Kautz, Jan and Hatamizadeh, Ali},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  eprint    = {2412.06464},
  archivePrefix = {arXiv},
  note      = {Table 3, 1.3B / 100B FineWeb-Edu: GDN Wiki 16.42 / LMB 12.17; Mamba2 16.56/12.56; DeltaNet 17.71/16.88; Transformer++ 18.53/18.32; Samba 16.13/13.29}
}
@article{hatamizadeh2026gdn2,
  title   = {Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention},
  author  = {Hatamizadeh, Ali and Choi, Yejin and Kautz, Jan},
  journal = {arXiv preprint arXiv:2605.22791},
  year    = {2026},
  note    = {Table 2, 1.3B / 100B FineWeb-Edu / 4k: GDN-2 Wiki 15.90 / LMB 11.41 / avg 53.11; KDA 16.81; GDN 16.40; Mamba-3 (MIMO) 16.45; Mamba-2 16.79. Per-layer recurrent state 262,144 floats (H=16, d_k=d_v=128, d_model=2048)}
}
@article{sun2024ttt,
  title   = {Learning to (Learn at Test Time): {RNN}s with Expressive Hidden States},
  author  = {Sun, Yu and Li, Xinhao and Dalal, Karan and Xu, Jiarui and Vikram, Arjun and Zhang, Genghan and Dubois, Yann and Chen, Xinlei and Wang, Xiaolong and Koyejo, Sanmi and Hashimoto, Tatsunori and Guestrin, Carlos},
  journal = {arXiv preprint arXiv:2407.04620},
  year    = {2024},
  note    = {Pile (2k/8k) and Books3 (1k-32k), 125M-1.3B, Chinchilla recipe. Figures 11-12. No enwik8 / WT-103 / PG-19}
}
@article{behrouz2025titans,
  title   = {Titans: Learning to Memorize at Test Time},
  author  = {Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal = {arXiv preprint arXiv:2501.00663},
  year    = {2025},
  note    = {Table 1: 340M/400M on 15B FineWeb-Edu, 760M on 30B. Titans LMM Wiki 26.18 / 25.03 / 20.04. No enwik8 / PG-19}
}
@inproceedings{sun2021longrange,
  title     = {Do Long-Range Language Models Actually Use Long-Range Context?},
  author    = {Sun, Simeng and Krishna, Kalpesh and Mattarella-Micke, Andrew and Iyyer, Mohit},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages     = {807--822}, year = {2021}, eprint = {2109.09115}, archivePrefix = {arXiv},
  note      = {THE retention/revisit slice convention: buckets by distance-to-last-occurrence (>2K away, never-in-prefix), frequency, first-vs-rest subword, document type; 220K PG-19 validation tokens, 10-token target chunks}
}
@inproceedings{khandelwal2018sharp,
  title     = {Sharp Nearby, Fuzzy Far Away: How Neural Language Models Use Context},
  author    = {Khandelwal, Urvashi and He, He and Qi, Peng and Jurafsky, Dan},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2018}, url = {https://aclanthology.org/P18-1027/},
  note      = {Context perturbation by distance; effective context ~200 tokens; word order matters only within ~50}
}
@inproceedings{toner2024linear,
  title     = {An Analysis of Linear Time Series Forecasting Models},
  author    = {Toner, William and Darlow, Luke},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  series    = {PMLR}, volume = {235}, year = {2024}, eprint = {2403.14587}, archivePrefix = {arXiv},
  note      = {Table 2. DLinear/FITS/RLinear/NLinear are equivalent to unconstrained linear regression; closed-form OLS beats the SGD-trained variants in 23/32 settings}
}
@inproceedings{zeng2023dlinear,
  title     = {Are Transformers Effective for Time Series Forecasting?},
  author    = {Zeng, Ailing and Chen, Muxi and Zhang, Lei and Xu, Qiang},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2023}, eprint = {2205.13504}, archivePrefix = {arXiv},
  note      = {LTSF-Linear beats FEDformer by 20-50% on multivariate forecasting}
}
@inproceedings{kim2022rigorous,
  title     = {Towards a Rigorous Evaluation of Time-series Anomaly Detection},
  author    = {Kim, Siwon and Choi, Kukjin and Choi, Hyun-Soo and Lee, Byunghan and Yoon, Sungroh},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2022}, eprint = {2109.05257}, archivePrefix = {arXiv},
  note      = {Point-adjust protocol: a random anomaly score becomes state of the art}
}
@article{kratzert2019ealstm,
  title   = {Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets},
  author  = {Kratzert, Frederik and Klotz, Daniel and Shalev, Guy and Klambauer, G{\"u}nter and Hochreiter, Sepp and Nearing, Grey},
  journal = {Hydrology and Earth System Sciences}, volume = {23}, pages = {5089--5110}, year = {2019},
  doi     = {10.5194/hess-23-5089-2019},
  note    = {Table 3, 447 basins: EA-LSTM ensemble median NSE 0.74; LSTM+static 0.72; mHM regional 0.53; VIC regional 0.31. Train 1999-10-01..2008-09-30, test 1989-10-01..1999-09-30}
}
@article{lees2022concept,
  title   = {Hydrological concept formation inside long short-term memory ({LSTM}) networks},
  author  = {Lees, Thomas and Reece, Steven and Kratzert, Frederik and Klotz, Daniel and Gauch, Martin and De Bruijn, Jens and Kumar Sahu, Reetik and Greve, Peter and Slater, Louise and Dadson, Simon J.},
  journal = {Hydrology and Earth System Sciences}, volume = {26}, pages = {3079--3101}, year = {2022},
  doi     = {10.5194/hess-26-3079-2022},
  note    = {Linear probes recover soil moisture and snow water equivalent from the LSTM cell state}
}
@article{addor2017camels,
  title   = {The {CAMELS} data set: catchment attributes and meteorology for large-sample studies},
  author  = {Addor, Nans and Newman, Andrew J. and Mizukami, Naoki and Clark, Martyn P.},
  journal = {Hydrology and Earth System Sciences}, volume = {21}, pages = {5293--5313}, year = {2017},
  doi     = {10.5194/hess-21-5293-2017},
  note    = {671 CONUS basins, 4-25000 km2, daily forcings + discharge + 27 static attributes}
}
@article{kratzert2023caravan,
  title   = {Caravan - A global community dataset for large-sample hydrology},
  author  = {Kratzert, Frederik and Nearing, Grey and Addor, Nans and Erickson, Tyler and Gauch, Martin and Gilon, Oren and Gudmundsson, Lukas and Hassidim, Avinatan and Klotz, Daniel and Nevo, Sella and Shalev, Guy and Matias, Yossi},
  journal = {Scientific Data}, volume = {10}, pages = {61}, year = {2023},
  note    = {6830 basins core, 1981-2020 daily. LICENCE (CC-BY-4.0) SINGLE-SOURCED -- verify}
}
@article{ariaschao2021ncmapss,
  title   = {Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics},
  author  = {Arias Chao, Manuel and Kulkarni, Chetan and Goebel, Kai and Fink, Olga},
  journal = {Data}, volume = {6}, number = {1}, pages = {5}, year = {2021},
  note    = {N-CMAPSS. DS02: 20 channels, 9 units; train {2,5,10,16,18,20} / test {11,14,15}}
}
@inproceedings{wu2019graphwavenet,
  title     = {Graph {W}ave{N}et for Deep Spatial-Temporal Graph Modeling},
  author    = {Wu, Zonghan and Pan, Shirui and Long, Guodong and Jiang, Jing and Zhang, Chengqi},
  booktitle = {Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI)},
  year      = {2019}, eprint = {1906.00121}, archivePrefix = {arXiv},
  note      = {Table 2. METR-LA 60min MAE: HA 4.16, ARIMA 6.90, FC-LSTM 4.37, DCRNN 3.60, STGCN 4.59, GWNet 3.53. PEMS-BAY 60min MAE: HA 2.88, DCRNN 2.07, GWNet 1.95}
}
@inproceedings{ohana2024thewell,
  title     = {The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning},
  author    = {Ohana, Ruben and McCabe, Michael and Meyer, Lucas and Morel, Rudy and Agocs, Fruzsina and Beneitez, Miguel and Berger, Marsha and Burkhart, Blakesley and Dalziel, Stuart and Fielding, Drummond and Fortunato, Daniel and Goldberg, Jared and Hirashima, Keiya and Jiang, Yan-Fei and Kerswell, Rich and Maddu, Suryanarayana and Miller, Jonah and Mukhopadhyay, Payel and Nixon, Stefan and Shen, Jeff and Watteaux, Romain and R{\'e}galdo-Saint Blancard, Bruno and Rozet, Fran{\c c}ois and Parker, Liam and Cranmer, Miles and Ho, Shirley},
  booktitle = {Advances in Neural Information Processing Systems, Datasets and Benchmarks Track},
  year      = {2024}, eprint = {2412.00568}, archivePrefix = {arXiv},
  note      = {16 datasets, 15TB, 6.9GB-5.1TB each, CC BY 4.0. Tables 2-3. VRMSE normalised so predicting the field mean = 1.0; several datasets have all baselines > 1}
}
```
