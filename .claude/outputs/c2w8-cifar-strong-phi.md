# c2w8-cifar-strong-phi — experiment-engineer report

> ## ⛔⛔ DATED CURATOR AMENDMENT — 2026-08-10 (`doc-curator-c2w8-pass3-close-fold`, **[C2W8-CLOSE]**) — **THE INVERSION QUALIFIER: THIS REPORT'S "STRONG φ" GAIN DOES NOT TRANSPORT TO ADDRESSABILITY — IT INVERTS**
> **⛔ The body of this report is NOT edited and nothing in it is deleted** (C-3 dated-banner precedent). ⛔⛔ **§1's MANDATORY PROVENANCE SENTENCE STILL STANDS AND STILL TRAVELS — this banner is ADDED BESIDE IT, never in place of it.**
>
> **What this report banks (unchanged, and correct on its own axis):** Split-CIFAR-10 re-priced at strong φ reads **0.319 ± 0.005 ACC (5 seeds)** against the in-harness `pca` reference **0.161** (+0.158 ± 0.012 paired), `laundered = True` at **−0.014240 ± 0.003, 0/5 seeds positive**.
>
> **What was measured one pass later** (`c2w8p3-capture-strong-phi` §3.1, the **completed** gate on the census rig, Split-CIFAR-10, `d = 12`, 3 seeds, streams bit-identical across arms ⇒ **paired**; charter Add.11 **§A31.4**): the **same built-and-priced `simclr` encoder** (`loss_first = 5.512242317199707`, `phi_param_floats = 225 536` — bit-identical to the banked `encoder_price.json`) is the **address-WORST arm beyond 2 SE**. `simclr − randconv` **A1 = −0.1406 ± 0.0508 (0/3 seeds simclr better)** · `simclr − pca` **= −0.1276 ± 0.0589 (0/3)** · `randconv − pca` = +0.0130 ± 0.1017 (**tied**). And the **unfitted `randconv` control buys the address geometry for FREE** — it clears the registered GO rule at `simclr`'s own margin with **0 fit steps** (`c2w8p3-phi-geometry` §2). ⇒ ⭐⭐ **on this substrate task accuracy and address geometry are ANTI-CORRELATED**, measured on the store, not on the geometry.
>
> ⛔⛔ **THE BINDING CONSEQUENCE (§A31.4): *"strong φ"* MAY NEVER AGAIN BE USED AS ONE UNDIFFERENTIATED NOTION.** ✅ **Admissible form: *"strong φ **by the CL-accuracy metric that defines it here**"*** — and any quotation of this report's gain states that **the same encoder is the worst arm on addressability**. ⛔ **Add.10 §A29.5's *"φ is the binding constraint on the CL rig"* does NOT survive this measurement.**
>
> ⚠ **What this banner does NOT touch:** the ACC / BWT / byte-ledger rows themselves · N255 / N256 / N257 / N258's dispositions · the ±0.0007 *settle-equals-its-own-kNN* result, which pass 3 **reproduces at a third substrate** (**N276**). ⛔ **The pass-3 numbers are CIFAR at `d = 12` on the CENSUS rig; no pass-1/2 census number is their baseline and this report is not their baseline either** (Head ruling R1). ⛔ **Nothing here adjudicates the arm A vs arm B race** (§A30.1, UNADJUDICATED).
> **Registry:** `negative_results.md` **N277** (+ **N276 / N283**); ledger ⟲ **C2W8 PASS-3 + CLOSE** addendum; primer **§11.27 Record 35** and the dated update on **Record 30**. **Sources:** `c2w8p3-capture-strong-phi` §3.1/§8, `c2w8p3-phi-geometry` §2, charter **Add.11 §A31.3–§A31.4**, the **`[C2W8-CLOSE]`** §10 entry.

**Task + acceptance criterion:** re-price the banked Split-CIFAR-10 null at strong φ
(`randconv`/`convae`/`simclr`, `task1_only`, ≥3 seeds, kNN-in-φ launder + baseline table +
byte ledger with φ params on every arm) and report the registered N7 / N8 with sign, SE and
seed count. **Status: done.**

> **⭐ DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). Four items.**
> 1. ⭐ **N7 HITS, N8 MISSES — the registered "clean scope clause" outcome fired.** Split-CIFAR
>    was a null at frozen-PCA φ; re-priced at strong φ (`simclr`, `enc_steps=8000`,
>    `phi_dim=256`, 225 536 φ floats ledgered on **every** arm), it reads **0.319 ± 0.005 ACC
>    (5 seeds)** — **+0.158 ± 0.012 paired** over the PCA-φ reference row re-run in this same
>    harness, and **−0.014 ± 0.003 (0/5 seeds positive) BELOW its own kNN-in-φ launder.**
>    ⛔ **A re-price is not a benchmark entry and this does not retire the null** — it scopes it
>    to the feature space, which is exactly what w25 diagnosed.
> 2. ⛔⛔ **The sharpest laundering statement the program has produced: at every strong-φ arm the
>    CLU settle equals its own same-keys kNN to within ±0.0007** (`randconv` +0.0003 ± 0.0003,
>    `convae` −0.0007 ± 0.0007, `simclr` +0.0002 ± 0.0004), against **−0.048 ± 0.005 at the PCA
>    arm.** In a strong φ the damped settle *is* nearest-key indexing; the entire −0.014 deficit
>    against the ring-buffer launder is the **class-balanced buffer's admission policy**, not the
>    read. Sixth consecutive laundering confirmation.
> 3. ⭐ **`cl-encoder`'s arm-isolation table needs an amendment: "reconstruction bought nothing"
>    is a READ-OUT artifact.** At the measured read-out (`keep`/plain-PCA/L2/`d=256`) `convae`
>    reaches CLU **0.267 ± 0.006** and kNN-ring **0.277 ± 0.003**, i.e. **+0.054 over `randconv`**
>    — where `cl-encoder` §3 measured convae 0.238 *below* randconv 0.244 at its pre-sweep
>    defaults and concluded the reconstruction objective "bought nothing" (−0.006). The
>    objective/architecture decomposition in that table is not safe to quote.
> 4. ⚠ **`cl_baselines.ConvNet` (the CIFAR backbone) cannot run under `jax_enable_x64`** —
>    it builds weights at the JAX default dtype while `build_cl_stream` always hands it float32
>    images ⇒ `lax.conv_general_dilated requires arguments to have the same dtypes`. **No shipped
>    result is affected** (all real runs are x64-off); it means the `backbone="cnn"` path is
>    untestable inside the full suite. That file is outside this task's ownership — needs an owner.

---

## ⭐ DIAL DECLARATION (echoed, protocol §7 / C2 form) — before the first result

- **Dial:** none as a new claim. This is a **re-price of a banked null** on the
  isolation/lifetimes benchmark. ⛔ No paper number, no new benchmark entry.
- **Laundering control:** kNN-in-φ at matched memory (N89/CM-22(i)) on **every** cell, in **both**
  shipped forms (same-keys and class-balanced ring buffer), computed in the **same φ object** the
  arm uses. The strong φ made the launder stronger (0.226 → 0.333) — that is the control working.
- **Falsifies the re-price:** strong φ leaves ACC statistically unmoved from the banked null.
  **Outcome: the falsifier did NOT fire** — the lift is +0.158 ± 0.012 paired, 3/3 shared seeds
  above the registered +0.10, so the null's diagnosed cause (the feature space) is **confirmed**.
- **Does NOT falsify:** losing to replay (CLU 0.319 vs iCaRL 0.419 — never claimed under any
  outcome, CM-23(q)); a negative launder margin (that is N8's registered prediction, prior 0.15).
- **N94:** the CLU store runs **zero gradient steps**, so no write-step floor applies to it; the
  **baseline** arms run at the declared reduced `baseline_iters=150` and every baseline number
  here is labelled **non-promotable**, exactly as w25 labelled them.

---

## §1 — ⛔ THE MANDATORY PROVENANCE (PREREG-C2W8 §8) — the only form these numbers travel in

> **"Split-CIFAR was a null at frozen-PCA φ; re-priced at strong φ (`simclr` trunk,
> `enc_steps=8000`, `phi_dim=256`, 225 536 φ parameter floats ledgered on the store *and* on both
> laundering controls), it reads 0.319 ± 0.005 ACC (5 seeds) — above every rehearsal-free
> baseline and level with GDumb, **still 0.014 ± 0.003 below its own kNN-in-φ launder**, and 0.100
> below iCaRL."**

⛔ It is **never** quoted without that provenance, in any artifact, draft or table. A favourable X
does **not** retire the null — it **scopes** it to the feature space, which is what was diagnosed.

**CM-23(q) travel rule, discharged** (the Split-MNIST side, quoted here only for context, with all
three sentences in one paragraph): the w25 entry is **+0.510 over the rehearsal-free class**,
**−0.153 vs iCaRL**, and **−0.036 LAUNDERED**, with the Split-CIFAR-10 null as its scope clause and
the Head's Addendum-2 ruling that this **does not count as an external benchmark won**. ⛔ "+0.510"
never appears without "−0.036 laundered" in the same paragraph.

---

## Flag-provenance table (governs every number in this report)

| item | value |
|---|---|
| branch / base | **`c2w8-cifar-strong-phi`**, base local `main @ d70898b` (worktree `../CHLU-c2w8-phi`) |
| commits (4) | `934c096` φ param counting · `a49e186` cl-entry ledger + `--set` overrides · `b6b5043` tests · `61d40db` test x64 fix |
| venv | **main venv reused** (`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python`) — no worktree `uv sync`, no JAX drift. **JAX 0.9.0 / equinox 0.13.4 / optax 0.2.6** |
| dataset / protocol | **Split-CIFAR-10, reduced protocol** (`apply_cifar10`): 5 tasks × 2 classes, **Class-IL** (task identity NOT given at test), 1000 train / 500 test per task, `baseline_iters=150`, 3-layer CNN backbone (`cnn_channels=[16,32,32]`, `mlp_width=128`), `fisher_samples=100`, `tune_baselines=False`, `clu_steps=150`. ⚠ **never literature-comparable Split-CIFAR-10** |
| φ regime | **`task1_only` ONLY** (binding, `PREREG_CL_PHI`). ⛔ `generic_frozen` NOT RUN (declared, §NOT-RUNs); `online` remains the unrun stub |
| φ arms | `pca` (`phi_dim=32`, the reference row) · `randconv` · `convae` · `simclr`, the last three at **`phi_dim=256`, `enc_steps=8000`**, `enc_head="pca"`, `enc_l2_normalize=True`, `enc_channels=[32,64,128]`, `enc_pool=2`, `enc_groups=8`, `enc_batch=128`, `enc_lr=1e-3`, `enc_temperature=0.5`, `enc_proj_dim=64`, shipped augment defaults |
| φ fit pool | `n_fit_region=25000`, `n_fit_pool=6000` ⇒ **4966 task-1-class images**, disjoint from every stream item, unsupervised, frozen at end of task 1, never refit, **never trained through the store** (w20 law) |
| seeds | `simclr` **0,1,2,3,4** · `randconv` **0,1,2,3,4** · `convae` **0,1,2** · `pca` **0,1,2** · baselines **0,1,2** |
| ⭐ stream identity | **the stream is bit-identical across arms at a given seed** — verified by SHA-256 of `train_X/train_y/test_X/test_y/fit_pool` (`318a2b8590716f68` for both the `pca` and the `simclr` config) ⇒ N7 is reported **PAIRED** |
| memory | **200 items** matched across the store, ER, DER++, iCaRL, GDumb and both kNN-in-φ launders |
| store | designed `AtomStorePotential(addr_dim=phi_dim)` + MVC-0 `Controller(allow_relocation=False)`, `clu_s_frac=0.2`, `d_safe_mult=4.4`, `s_policy="refit"`, `clu_b=1.0`, `clu_alpha=1e-3`, `clu_gamma=0.1`, `clu_steps=150`, `dt` auto, `clu_tail_frac=0.1`, **`newtonian_identity`**, `rollout_chunk=256`, `count_phi_param_floats=True` |
| items run | **`--items entry` only** (no retry, no retention, no frontier) |
| commands | `python -m chlu.experiments.exp_cl_entry --dataset cifar10 --items entry [--baselines none] --set phi_arm=… --set phi_dim=… --set enc_steps=8000 --set phi_regimes=task1_only --set n_fit_region=25000 --set n_fit_pool=6000 --set seeds=…` |
| PREREG | `.claude/outputs/c2w8-cifar-strong-phi/PREREG.md`, written **after the pricing probe and before the first measuring cell**; inherits `PREREG-C2W8.md` §6 N7/N8 unchanged |
| artifacts | `results/{pca,randconv,convae,simclr}_metrics.json` + `_run.log`, `encoder_price.json`, `summary.json`, `render_output.txt`; every number re-derived by `render.py` |

---

## 1. The four arms (Split-CIFAR-10 reduced protocol, Class-IL, `task1_only`, 200 items)

| arm | `phi_dim` | fit steps | n | **CLU ACC** | kNN-φ same keys | kNN-φ ring buffer | BWT | forgetting |
|---|---|---|---|---|---|---|---|---|
| `pca` (**the banked-null reference row**) | 32 | — | 3 | **0.161 ± 0.004** | 0.209 ± 0.009 | 0.226 ± 0.005 | −0.144 ± 0.003 | 0.151 ± 0.005 |
| `randconv` (**architecture only, sees no data**) | 256 | 0 | 5 | **0.213 ± 0.006** | 0.213 ± 0.007 | 0.235 ± 0.005 | −0.222 ± 0.005 | 0.222 ± 0.005 |
| `convae` (reconstruction) | 256 | 8000 | 3 | **0.267 ± 0.006** | 0.268 ± 0.005 | 0.277 ± 0.003 | −0.235 ± 0.008 | 0.235 ± 0.008 |
| ⭐ **`simclr` (contrastive) — the headline arm** | 256 | 8000 | 5 | **0.319 ± 0.005** | 0.319 ± 0.005 | **0.333 ± 0.006** | −0.243 ± 0.006 | 0.243 ± 0.006 |

(± is the **standard error** over seeds. Per-seed CLU ACC — `simclr`: 0.3256 / 0.3304 / 0.3028 /
0.3128 / 0.3240; `randconv`: 0.2224 / 0.2160 / 0.2292 / 0.2020 / 0.1952; `convae`: 0.2704 / 0.2560 /
0.2748; `pca`: 0.1576 / 0.1564 / 0.1684.)

**Harness-validity evidence (three independent reproductions):**
- the `pca` reference row **0.161 ± 0.004** reproduces the banked w25 null **0.149 ± 0.013**
  (within ~1 SE; the small positive offset is the larger task-1 fit pool, 4966 vs w25's `n_fit_pool
  =3000` region);
- the `simclr` ring-buffer launder **0.333 ± 0.006** reproduces `cl-encoder`'s independently
  measured 8 k gate value **0.339 ± 0.015**;
- the baseline table (§3) reproduces w25's CIFAR values to ≤0.017 on every method.

## 2. ⭐ The two registered readings

### N7 — does strong φ lift CLU ACC by ≥ +0.10 over the banked PCA-φ null? **HIT (on `simclr`).**

Reported **paired** on the three shared seeds (the stream is bit-identical across arms) *and*
unpaired against the banked value:

| arm | paired vs the in-harness `pca` row (seeds 0,1,2) | per-seed | vs the banked 0.149 | verdict vs the registered **+0.10** |
|---|---|---|---|---|
| `randconv` | **+0.0617 ± 0.0016** (3/3 positive) | +0.065 / +0.060 / +0.061 | +0.064 | ⛔ **MISS** |
| `convae` | **+0.1063 ± 0.0038** (3/3) | +0.113 / +0.100 / +0.106 | +0.118 | ◐ **HIT, marginally** — the mean clears by +0.006 ± 0.004 (1.7 SE) and one seed (0.0996) sits below the threshold |
| ⭐ **`simclr`** | **+0.1588 ± 0.0123** (3/3) | +0.168 / +0.174 / +0.134 | **+0.170** | ✅ **HIT**, every seed individually above +0.10 |

⇒ **N7 = HIT.** The registered prior was 0.55; my PREREG raised it to 0.80 and registered
`randconv`/`convae` as misses — **`randconv` missed as registered, `convae` did not** (see
reconciliation item 3: the miss was predicted off `cl-encoder`'s pre-sweep read-out corner).

### N8 — does CLU beat its **own** kNN-in-φ launder? ⛔ **MISS on every arm. The store is LAUNDERED.**

Paired per seed, against the **stronger** of the two launder lines (the ring buffer, as
`entry_verdict` does):

| arm | **N8 = CLU − best launder** | seeds positive | vs same-keys kNN | vs ring-buffer kNN |
|---|---|---|---|---|
| `pca` | **−0.068 ± 0.006** | 0/3 | −0.0480 ± 0.0050 | −0.0653 ± 0.0085 |
| `randconv` | **−0.024 ± 0.007** | 1/5 | **+0.0003 ± 0.0003** | −0.0224 ± 0.0081 |
| `convae` | **−0.012 ± 0.006** | 0/3 | **−0.0007 ± 0.0007** | −0.0104 ± 0.0070 |
| ⭐ **`simclr`** | **−0.014 ± 0.003** | **0/5** | **+0.0002 ± 0.0004** | −0.0142 ± 0.0035 |

(N8 per seed at `simclr`: −0.0240 / −0.0148 / −0.0184 / −0.0104 / −0.0036.)

⇒ **N8 = MISS, 4.7 SE below zero, 0/5 seeds.** The registered prior was 0.15 and it holds. Two
things worth carrying:

1. ⭐ **The deficit SHRINKS monotonically as φ gets stronger** (−0.068 → −0.024 → −0.012/−0.014),
   i.e. most of the w25 CIFAR laundering gap was φ's weakness, not the store's — but it does not
   cross, and the residual is stable at ≈ −0.014 with a tight SE.
2. ⛔⛔ **Decomposed, the residual is entirely the ring buffer's class-balancing, not the read.**
   Against the *same keys* the settle is a **tie to four decimals** on all three strong arms
   (+0.0002 ± 0.0004 at `simclr`), where at the PCA arm the settle was **0.048 below** the same
   keys. In a strong φ the damped-Verlet settle *is* nearest-key indexing. **This is the cleanest
   statement of the laundering result the program has: in a good address space the store's read
   adds exactly nothing over `argmin‖φ(q) − key‖`, and the remaining 0.014 is bought by a
   class-balanced admission policy any ring buffer can run.**

⭐ **This is the registered "N7 without N8" outcome, and per the task it is a clean, publishable
scope clause, not a disappointment:** the feature space **was** the null's cause, **and** the store
still adds nothing over the trivial substitute in the better space.

## 3. The mandatory baseline table (3 seeds, φ-independent, run once at the `pca` arm)

Sorted by ACC; **the byte ledger carries φ params on the CLU/launder arms and fixed backbone state
on the gradient arms** (§4). ⚠ every baseline is **non-promotable** (reduced `baseline_iters=150`).

| method | class | **ACC** | BWT | forgetting | mem floats | fixed/φ floats | **total floats** |
|---|---|---|---|---|---|---|---|
| joint (offline) | upper bound | 0.474 ± 0.015 | −0.093 | 0.144 | 0 | 81 290 | 81 290 |
| **iCaRL** | replay | **0.419 ± 0.012** | −0.231 | 0.231 | 614 600 | 81 290 | 695 890 |
| **ER** | replay | 0.367 ± 0.003 | −0.452 | 0.452 | 614 600 | 81 290 | 695 890 |
| **DER++** | replay | 0.367 ± 0.009 | −0.501 | 0.501 | 616 600 | 81 290 | 697 890 |
| ⭐ **CLU entry, `simclr` φ (5 seeds)** | rehearsal-free | **0.319 ± 0.005** | −0.243 | 0.243 | 53 000 | **225 536 (φ)** | **278 536** |
| ⛔ **kNN-in-φ ring buffer, `simclr` φ** | **the launder** | **0.333 ± 0.006** | −0.229 | 0.229 | 51 200 | **225 536 (φ)** | **276 736** |
| **GDumb** (matched memory) | replay | 0.318 ± 0.029 | −0.220 | 0.234 | 614 600 | 81 290 | 695 890 |
| CLU entry, `convae` φ (3 seeds) | rehearsal-free | 0.267 ± 0.006 | −0.235 | 0.235 | 53 000 | 225 536 (φ) | 278 536 |
| kNN-in-φ ring buffer, `pca` φ | launder | 0.226 ± 0.007 | −0.233 | 0.234 | 6 400 | 101 376 (φ) | 107 776 |
| CLU entry, `randconv` φ (5 seeds) | rehearsal-free | 0.213 ± 0.006 | −0.222 | 0.222 | 52 735 | 225 536 (φ) | 278 271 |
| kNN-in-φ same keys, `pca` φ | launder | 0.209 ± 0.012 | −0.198 | 0.198 | 8 200 | 101 376 (φ) | 109 576 |
| CLU entry, `pca` φ (reference row) | rehearsal-free | 0.161 ± 0.005 | −0.144 | 0.151 | 8 200 | 101 376 (φ) | 109 576 |
| **finetune** | rehearsal-free (the null) | 0.160 ± 0.001 | −0.776 | 0.776 | 0 | 81 290 | 81 290 |
| **LwF** | rehearsal-free (known null) | 0.159 ± 0.002 | −0.773 | 0.773 | 0 | 162 580 | 162 580 |
| **SI** | rehearsal-free (known null) | 0.157 ± 0.002 | −0.763 | 0.763 | 0 | 243 870 | 243 870 |
| **EWC** | rehearsal-free (known null) | 0.156 ± 0.000 | −0.682 | 0.682 | 0 | 243 870 | 243 870 |

**The three sentences, computed** (CLU = the `simclr` arm; ⛔ none may travel without §1's
provenance):
1. `wins_rehearsal_free_class = True`, margin **+0.159** over the best rehearsal-free baseline
   (finetune 0.160). ⚠ **The margin is against the known null** — EWC/SI/LwF collapse in Class-IL
   *by construction* (van de Ven & Tolias); their collapse is **not** a CLU achievement.
2. `deficit_vs_replay = −0.100` (vs iCaRL 0.419); **level with GDumb** (+0.001, inside GDumb's own
   ±0.029). ⛔ **"Beats replay" is not claimed and is not true.**
3. ⛔ `laundered = True`, `clu_minus_launder = −0.014 ± 0.003` (0/5 seeds). **The win is φ's and
   the buffer's, not the store's** (N89/CM-22(i), required wording).

**Where the mechanism still shows, and where it stopped showing.** Against the parametric
rehearsal-free class the store's BWT is **−0.243 vs −0.68…−0.78** — still a 3× smaller backward
loss — and it is level with **iCaRL's −0.231** at **2.50× fewer total bytes**. But ⚠ **the
forgetting advantage shrank as φ improved**: −0.144 (pca) → −0.243 (simclr). The w25 sentence
*"the store does not forget; it never knew"* is now literally quantified — the store forgets
**more** once its addresses actually carry class information, so the forgetting axis (`matched-
bytes-frontier`'s claim) must be **re-measured at the strong φ before it is quoted on CIFAR.**

## 4. ⭐ The byte ledger, with φ params counted on **every** arm (charter §A4.3)

Measured from the fitted φ object, not from a formula (`read_in_param_floats`; the conv trunk
93 696 + the `h→φ` PCA head 131 840 = **225 536** floats; the SimCLR projection head and the
convae decoder are discarded after fitting and are **not** counted):

| arm | store items | store floats | **φ param floats** | **CLU total** | **launder total** |
|---|---|---|---|---|---|
| `pca`-32 | 200 | 8 200 | **101 376** | **109 576** | 107 776 |
| `randconv` / `convae` / `simclr` (`phi_dim=256`) | 199–200 | 52 735–53 000 | **225 536** | **278 536** | 276 736 |

- ⭐ **The strong φ is 4.25× the store's own memory.** A strong-φ CLU arm is *dominated by its
  encoder*, and any "N floats per item" sentence about CIFAR that omits the encoder is wrong by a
  factor of five. (`cl-encoder`'s "an address is 256 floats vs a raw exemplar's 3072" is the
  per-item cost only.)
- **It does not change the N8 comparison, by design:** the launder reads through the *same* φ and
  carries the *same* 225 536 floats (pytest-asserted).
- **It does not overturn the replay comparison either:** CLU **278 536** total vs ER/iCaRL/GDumb
  **695 890** ⇒ CLU is still **2.50× cheaper in bytes** while scoring 0.100 below iCaRL and level
  with GDumb. (PREREG predicted 278 500 and 2.5×.)
- ⚠ The gradient baselines' ledger is *their* answer to φ: EWC/SI carry **243 870** (backbone ×3),
  LwF **162 580** (×2) — all above the `pca`-φ arm's 109 576 and below the strong-φ arm's 278 536.

## 5. Geometry (end of stream, 200 live wells) — a better φ does NOT de-crowd the store

| arm | median-NN | `s` | `σ_q` (norm) | **corrected packing slack** |
|---|---|---|---|---|
| `pca` | 9.551 | 1.886 | 9.038 | **0.341** |
| `randconv` | 0.852 | 0.170 | 0.830 | **0.331** |
| `convae` | 1.005 | 0.198 | 0.979 | **0.331** |
| `simclr` | 0.975 | 0.192 | 0.946 | **0.332** |

⛔ **Never quote the retracted 1.08.** The slack is **0.331–0.341 on all four arms**, i.e.
identical to w25's CIFAR 0.337–0.345 and MNIST ≈0.33 and to `cl-encoder`'s 0.332 — **P8 confirmed
a second time, now with the CLU settle actually run**: crowding is intrinsic to a *classification*
stream (`σ_q ≈ median-NN`: a test query is about as far from its nearest address as two addresses
are from each other), not a φ defect. This is the geometric reason item 2 above holds: with the
basins that crowded, the settle can only land in the nearest well.

## 6. PREREG scorecard (`.claude/outputs/c2w8-cifar-strong-phi/PREREG.md`)

| # | registered before the sweep | measured | verdict |
|---|---|---|---|
| N7 (inherited) | lift ≥ +0.10 over the banked PCA-φ null; my P(hit) = 0.80 | **+0.159 ± 0.012 paired** at `simclr` | ✅ **HIT** |
| N8 (inherited) | CLU beats its own launder; prior 0.15. My point prediction **−0.04**, band −0.08…−0.01 | **−0.014 ± 0.003**, 0/5 seeds | ✅ **MISS as registered**, and inside the registered band |
| arm `randconv` | CLU 0.19–0.25, N7 MISS | **0.213 ± 0.006**, +0.062 | ✅ both halves |
| arm `convae` | CLU 0.18–0.25, N7 MISS | **0.267 ± 0.006**, **+0.106 ⇒ HIT** | ❌ **falsified** — see reconciliation item 3 (the prediction was read off `cl-encoder`'s pre-sweep read-out corner; the objective *does* buy +0.054 over `randconv` at the measured read-out) |
| arm `simclr` | CLU 0.28–0.34 | **0.319 ± 0.005** | ✅ mid-band |
| φ ledger | 225 536 φ floats = 4.3× the store; CLU total ≈ 278 500; ≈2.5× cheaper than ER | **225 536**, 4.25×, **278 536**, **2.50×** | ✅ exact |
| falsifier | lift < +0.05 ⇒ the null re-prices to the *discipline* | did not fire | ✅ feature-space diagnosis **confirmed** |

**Score: 6 ✅ / 1 ❌.** The single failure (`convae`) is the informative one and is the third
reconciliation item.

## 7. ⛔ DECLARED NOT-RUNs (declared in the PREREG **before** the sweep; **never** nulls)

1. **`simclr` @ `enc_steps=20000`** — `cl-encoder`'s gate-clearing arm. Priced at **64–75 min/seed**
   ⇒ ≈5.3 h for 5 seeds, which does not fit beside four arms in this budget. ⇒ ⚠ **the headline
   arm here is the UNDER-TRAINED one** (NT-Xent 4.07 at 8 k vs 3.98 at 20 k vs a converged ~2.5–3),
   and `cl-encoder`'s 200-item launder rose 0.339 → 0.357 across that gap. **Every N7/N8 number in
   this report carries `enc_steps=8000`.** N7 would only get *safer* at 20 k; N8's direction is
   unknown but the arm-trend here (deficit −0.024 → −0.012 → −0.014) is not obviously closing.
2. **`generic_frozen`** — a second φ fit per seed (doubles the cost) and a declared upper bound that
   may never be a headline (w24 ruling). ⇒ **no strict-φ cost is measured here**; `cl-encoder` §5's
   **+0.020 ± 0.010** at this exact arm stands unamended.
3. **MNIST at the strong φ** — out of scope; w25's MNIST numbers stand unchanged.
4. **`online` φ** — out of protocol, remains the unrun stub.
5. **retry / retention / matched-bytes frontier at the strong φ** — `--items entry` only.
   ⚠ The frontier is the one that most needs re-running (§3's BWT collapse).
6. **≥5 seeds on `convae` and `pca`** — 3 seeds each; the arm cut was declared in the PREREG and
   seeds were held at ≥3 everywhere (task §Honesty: cut arms, not seeds).

## 8. How I verified

- **Encoder pricing before the sweep** (`price_encoder.py`, `results/encoder_price.json`):
  `randconv` 2.9 s, `convae` 0.205 s/step, `simclr` 0.226 s/step (both including the ~3–4 s JIT
  compile amortised over a 50-step probe; netted, this reproduces `cl-encoder`'s 21 min @ 8 000
  steps). Actual wall-clock: `randconv` 5 seeds **7 min**; `pca` 3 seeds + 9 baselines **37 min**;
  `convae` 3 seeds **1 h 51 m**; `simclr` 5 seeds **2 h 47 m** (the last two overlapped, hence
  above the ≈1 h/1 h 50 m projections). **Total measured-run budget ≈ 3 h wall-clock.**
- **New tests:** `pytest tests/test_cifar_strong_phi.py -q -p no:randomly --no-cov` → **10 passed**
  (31.8 s), and **10 passed** again under `JAX_ENABLE_X64=1` (30.3 s).
- **Full suite** on the branch: first run **1418 passed / 2 failed** (3270 s) — both failures were
  my own new e2e cells hitting the pre-existing x64 CNN-backbone hazard (reconciliation item 4);
  fixed in `61d40db` and re-run: see §"Suite" below.
  **Count arithmetic: handover base = 1410 passed at `d70898b`; this branch adds 10 tests ⇒ 1420
  expected**, and 1418 + 2 = **1420 collected** on the first run, which is the arithmetic check.
- `ruff check chlu/` → **All checks passed**.
- **Stream-identity check** (so N7 may be paired): SHA-256 over
  `train_X/train_y/test_X/test_y/fit_pool_task1_only` is `318a2b8590716f68` for **both** the `pca`
  and the `simclr` configurations at seed 0 ⇒ the arms differ only in φ.
- Every number above is re-derived from the shipped JSONs by `render.py`
  (`results/render_output.txt`, `results/summary.json`).

- ⭐ **Suite result (final, on `61d40db`, machine otherwise idle):**
  `pytest tests/ -q -p no:randomly --no-cov` → **`1420 passed, 0 failed` (1880.24 s = 31 m 20 s)**,
  log at `results/full_suite2.log`. **Count arithmetic: 1410 (handover base at `d70898b`) + 10 new
  = 1420. ✅ exact.**

## 9. Known limitations (stated, not hidden)

1. **The headline arm is under-trained** (`enc_steps=8000`, NOT-RUN 1). Every N7/N8 number carries
   that flag.
2. **The reduced CIFAR protocol** (1000 train / 500 test per task, 150 optimizer steps, 3-layer
   CNN) — internally matched, **never** literature-comparable, and every baseline is
   **non-promotable** under N94.
3. **`convae`/`pca` are 3 seeds**, `simclr`/`randconv` 5. The headline margin (N8) is on 5 seeds as
   required.
4. **The read-out config** (`enc_head="pca"`, `enc_l2_normalize=True`, `keep`-spatial) was selected
   by `cl-encoder` on **seed 0 of the `simclr` trunk** and applied unchanged here to all arms.
   `convae`'s N7 hit is therefore obtained under a read-out tuned on a different objective — which
   makes it a *conservative* result for `convae`, but it is selection on the decision metric one
   wave upstream and should be said.
5. **Baselines ran at the `pca` arm only** (they never see φ). Legitimate because the stream is
   bit-identical across arms — but they are 3 seeds against `simclr`'s 5.
6. **No CLU-side retrieval/retry number** at the strong φ, and **no frontier re-run** — §7 items 5.

---

## Git footprint

- **Branch `c2w8-cifar-strong-phi`**, base local `main @ d70898b`, worktree
  `/Users/user/Desktop/CHLU-c2w8-phi`. **No push, no PR.** Left for Hub review.
- **4 commits:** `934c096` (φ read-ins report their own parameter count) · `a49e186` (cl-entry:
  φ params on the byte ledger of every arm + `--set` overrides + `count_phi_param_floats`) ·
  `b6b5043` (10 tests) · `61d40db` (test x64 backbone fix).
- **Files — 1 new, 4 surgical, all inside the declared ownership:** NEW
  `tests/test_cifar_strong_phi.py` (10 tests). Modified `chlu/experiments/phi_encoders.py`
  (`module_param_floats`, `_PCAHead.param_floats`, `ConvEncoderReadIn.trunk`/`param_floats`,
  provenance field), `chlu/experiments/exp_phi_read_in.py` (`param_floats` on `PCAReadIn`/
  `AEReadIn`, `read_in_param_floats` dispatcher, one import), `chlu/experiments/exp_cl_entry.py`
  (φ ledger on the store/launder/baseline/frontier rows + table columns, `apply_overrides` +
  `--set`, config echo), `chlu/config.py` (**+1 field `count_phi_param_floats` appended at the very
  end of the CL block, immediately before `seed`** — no existing default touched).
- ⚠ **`chlu/config.py` adjacency conflict expected** with `c2w8-well-lifecycle` (wt1), which also
  appends there; resolve **additively** (w23/w24/w25 precedent). No other file is shared: I touched
  **none** of `controller.py`, `clu_system.py`, `friction_field.py`, `well_lifecycle.py`,
  `usage_telemetry.py`, `exp_well_lifecycle.py`, the C2W6 files or the C2W7 files.
- **Not touched, deliberately:** `chlu/experiments/cl_baselines.py` (not in my ownership) — hence
  the `fixed_state_floats` φ-term correction is applied *at the call site* in `exp_cl_entry.py`
  rather than in the formula, and the ConvNet x64 hazard is reported rather than patched.
- `git rebase main` = no-op (base unmoved at `d70898b`). Branch ref verified from the MAIN repo.
- **Worktree left in place** for the Hub to remove at integration (canonical copies of every JSON
  quoted here are already in `.claude/outputs/c2w8-cifar-strong-phi/results/`).

## Open questions / follow-ups / risks

1. ⭐⭐ **The decisive follow-up is NOT more φ compute — it is the ring buffer's class balancing.**
   The whole residual N8 deficit at `simclr` is the class-balanced admission policy (the settle
   ties the same-keys kNN to ±0.0004). The MVC-0 controller already *has* a class-balanced
   eviction rule; measuring why it under-performs a plain balanced ring buffer by 0.014 is a
   cheap, contained question and is the only visible route to crossing N8 on this benchmark.
2. ⚠ **The forgetting claim on CIFAR needs re-measuring at the strong φ.** CLU BWT went
   −0.144 → −0.243 as φ improved; `matched-bytes-frontier`'s forgetting axis was measured at the
   PCA φ, where the store "never knew" and therefore never forgot. **Do not quote a CIFAR
   forgetting advantage from the PCA-φ frontier at a strong φ without re-running it.**
3. **`enc_steps=20000` is the one unpriced lever left** (≈5.3 h for 5 seeds, no code change). It
   would move N7 further into the green and settle whether the −0.014 residual is fixed or
   closing.
4. **`convae` at 0.267 is 84 % of `simclr`'s lift for a cheaper, simpler objective**, and it
   contradicts `cl-encoder`'s arm decomposition. If the program wants a defensible "the objective
   matters" sentence it must re-run `cl-encoder` §3's isolation table at the measured read-out.
5. **Risk of over-reading N7.** The lift is real and pre-registered, but the same φ raises the
   *launder* by exactly as much (0.226 → 0.333). Nothing here is a store result; the store's own
   contribution measured against its trivial substitute is **0.000 ± 0.0004**.

---

## Proposed handover updates (for the Hub)

1. **§6 ground truth — new entry.** *The Split-CIFAR-10 null is RE-PRICED at strong φ (C2W8).*
   Split-CIFAR-10 reduced protocol, Class-IL, `task1_only`, 200 items, `phi_arm=simclr`,
   `phi_dim=256`, `enc_steps=8000`, 5 seeds: **CLU 0.319 ± 0.005 ACC / BWT −0.243**, **+0.159 over
   the best rehearsal-free baseline**, **−0.100 vs iCaRL**, **level with GDumb**, ⛔ **−0.014 ±
   0.003 BELOW its own kNN-in-φ launder (0/5 seeds) ⇒ LAUNDERED (sixth consecutive confirmation)**.
   ⛔ **A re-price is not a benchmark entry; it never travels without §1's provenance sentence and
   it does NOT retire the w25 null — it scopes it to the feature space.** The arm ladder is
   `pca` 0.161 → `randconv` 0.213 → `convae` 0.267 → `simclr` 0.319.
2. **Candidate N-entries (3).** (a) ⭐⭐ *"In a strong φ the CLU settle equals its own same-keys kNN
   to ±0.0004 (3 arms, 11 seeds), against −0.048 at the PCA φ; the entire residual laundering
   deficit is the control buffer's class balancing, not the read"* — tier A, the sharpest
   laundering statement the program holds, and it converts the launder from a mystery into a
   named, cheap mechanism question. (b) *"Strong φ lifts Split-CIFAR CLU ACC +0.159 ± 0.012
   (paired, 3/3 seeds) with zero change to the store or the stream discipline ⇒ w25's feature-space
   diagnosis is CONFIRMED and the null re-prices to the feature space, not the discipline"* —
   tier A, scope-defining. (c) ⚠ *"The store's CIFAR forgetting advantage is a property of a broken
   φ: BWT −0.144 (PCA) → −0.243 (SimCLR) as the addresses start carrying class information"* —
   tier A, and it puts a **hold** on quoting the matched-bytes forgetting frontier at a strong φ.
3. **§7 — new config/CLI surface (all additive, all defaults preserving):**
   `ExperimentClEntryConfig.count_phi_param_floats` (default `True`, appended at the end of the CL
   block); `exp_cl_entry --set KEY=VALUE` (repeatable, applied **after** `--quick`/`--dataset`);
   `exp_phi_read_in.read_in_param_floats(phi)` + `param_floats()` on `PCAReadIn`/`AEReadIn`/
   `ConvEncoderReadIn`/`_PCAHead`; `phi_encoders.module_param_floats`. The results JSON's
   `baseline_table` gains `phi_param_floats` / `fixed_state_floats` / `total_floats` columns.
4. **§7 — the `apply_cifar10` clobber trap is now CLOSED** (`cl-encoder` handover item 8): overrides
   are applied last and echoed at launch. The trap itself is pytest-asserted so it cannot silently
   return.
5. **⚠ §7 — NEW open issue, no owner:** `cl_baselines.ConvNet` (`backbone="cnn"`) raises
   `lax.conv_general_dilated requires arguments to have the same dtypes` under `jax_enable_x64`
   because `build_cl_stream` pins the stream to float32 while the module builds at the JAX default
   dtype. **No shipped result is affected** (real runs are x64-off) but the CIFAR backbone is
   untestable inside the full suite. Same family as N211/§7.23. One-line fix in a file this task
   does not own.
6. **⚠ `cl-encoder` §3's arm-isolation table needs a dated amendment.** "Reconstruction bought
   nothing (−0.006 vs `randconv`)" was measured at the pre-sweep read-out corner; at the shipped
   measured read-out `convae` beats `randconv` by **+0.054** and clears the +0.10 N7 threshold.
7. **Do-not-quote list.** ⛔ any strong-φ CIFAR number without `enc_steps=8000` attached (the
   gate-clearing 20 k arm is a **declared NOT-RUN** here). ⛔ any φ-arm number without the §1
   provenance sentence. ⛔ "the encoder improved CLU" — the encoder improved *the address space*,
   and it improved the launder by the same amount. ⛔ any CIFAR "N floats per item" sentence that
   omits the **225 536** φ floats. ⛔ the retracted packing slack 1.08 (measured 0.331–0.341).
   ⛔ a CIFAR forgetting advantage carried over from the PCA-φ frontier.
8. **Test count:** +10 ⇒ integration should expect **1420** (base 1410).
9. **Compute note for wave planning:** ≈3 h wall-clock of measured runs, of which ~2.5 h is φ
   fitting. `randconv` and the `pca` reference are effectively free; a 20 k-step `simclr` arm is
   ≈5.3 h for 5 seeds.
