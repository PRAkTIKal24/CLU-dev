import re, sys, hashlib

P = "/Users/user/Desktop/V5_PALM_Submission/paper.tex"
src = open(P, encoding="utf-8").read()
orig = src

# (label, old, new, expected_count)
REPL = [
# ---- CITATION hunks ----
("C01","canonical Blelloch \\& Golovin (2007) placement",
       "canonical \\citet{blelloch_strongly_2007} placement",1),
("C02","(Yang, 2026; Uddin et al., 2026)",
       "\\citep{yang_control-plane_2026,uddin_recall_2026}",1),
("C03","(Rasmussen et al., 2025; Chhikara et al., 2025)",
       "\\citep{rasmussen_zep_2025,chhikara_mem0_2025}",1),
("C04","(Chakraborttii et al., 2026; Wang \\& Zhang, 2026)",
       "\\citep{chakraborttii_ghost_2026,wang_memleak_2026}",1),
("C05","introduced as CHLU by Jawahar \\& Pierini (2026)",
       "introduced as CHLU by \\citet{jawahar_chlu_2026}",1),
("C06","Utilizing Blelloch \\& Golovin's (2007) stable-matching",
       "Utilizing \\citeauthor{blelloch_strongly_2007}'s \\citeyearpar{blelloch_strongly_2007} stable-matching",1),
("C07","(e.g., MemGPT, Packer et al., 2023; Mem0, Chhikara et al., 2025)",
       "(e.g., MemGPT, \\citealp{packer_memgpt_2024}; Mem0, \\citealp{chhikara_mem0_2025})",1),
("C08","(e.g., Infini-attention, Munkhdalai et al., 2024)",
       "(e.g., Infini-attention, \\citealp{munkhdalai_leave_2024})",1),
("C09","(Park et al., 2023)","\\citep{park_generative_2023}",1),
("C10","(Zhong et al., 2024)","\\citep{zhong_memorybank_2023}",1),
("C11","(Sukhbaatar et al., 2021)","\\citep{sukhbaatar_not_2021}",1),
("C12","(Behrouz et al., 2025)","\\citep{behrouz_titans_2024}",2),
("C13","(Rasmussen et al., 2025)","\\citep{rasmussen_zep_2025}",1),
("C14","(Chakraborttii et al., 2026)","\\citep{chakraborttii_ghost_2026}",1),
("C15","(Yang, 2026)","\\citep{yang_control-plane_2026}",1),
("C16","(Uddin et al., 2026)","\\citep{uddin_recall_2026}",1),
("C17","removal is Guo et al.'s (2020) Sec.~2",
       "removal is \\citeauthor{guo_certified_2023}'s \\citeyearpar{guo_certified_2023} Sec.~2",1),
("C18","(Bourtoule et al., 2021; Ginart et al., 2019; Sekhari et al., 2021)",
       "\\citep{bourtoule_machine_2019,ginart_making_2019,sekhari_remember_2021}",1),
("C19","(Blelloch \\& Golovin, 2007)","\\citep{blelloch_strongly_2007}",1),
("C20","(Hochreiter \\& Schmidhuber, 1997)","\\citep{hochreiter_long_1997}",1),
("C21","(Aitken et al., 2022)","\\citep{aitken_geometry_2022}",1),
("C22","(Minami \\& Hidaka, 2018)","\\citep{minami_spontaneous_2018}",1),
("C23","(arXiv:2605.03338)","\\citep{mo_symmetry-protected_2026}",1),
("C24","date to Snyder (1977) and Andersson \\& Ottmann (1995)",
       "date to \\citet{snyder_uniquely_1977} and \\citet{andersson_new_1995}",1),
("C25","is Micciancio's (1997)",
       "is \\citeauthor{micciancio_oblivious_1997}'s \\citeyearpar{micciancio_oblivious_1997}",1),
("C26","are Naor \\& Teague's (2001)",
       "are \\citeauthor{naor_anti-persistence_2001}'s \\citeyearpar{naor_anti-persistence_2001}",1),
("C27","is Blelloch \\& Golovin's (2007), whose table",
       "is \\citeauthor{blelloch_strongly_2007}'s \\citeyearpar{blelloch_strongly_2007}, whose table",1),
("C28","Blelloch, Golovin \\& Vassilevska (2008) had already",
       "\\citet{hutchison_uniquely_2008} had already",1),
("C29","(Buchbinder \\& Petrank, 2003)","\\citep{goos_lower_2003}",1),
("C30","No: Blelloch \\& Golovin (2007) own it outright, and unique representation in a geometric setting is also taken (2008).",
       "No: \\citet{blelloch_strongly_2007} own it outright, and unique representation in a geometric setting is also taken \\citeyearpar{hutchison_uniquely_2008}.",1),
]

for lab, old, new, cnt in REPL:
    got = src.count(old)
    if got != cnt:
        sys.exit(f"ABORT {lab}: expected {cnt} occurrence(s), found {got}\n  OLD: {old!r}")
    src = src.replace(old, new)
    print(f"  {lab}: {cnt} replacement(s) OK")

# ---- BIBLIOGRAPHY hunk ----
m = re.search(r"\\section\*\{References\}\n\\begin\{itemize\}\n(.*?)\n\\end\{itemize\}\n",
              src, re.S)
assert m, "bibliography block not found"
items = [l for l in m.group(1).split("\n") if l.startswith("\\item")]
assert len(items) == 31, f"expected 31 items, got {len(items)}"
jude = [l for l in items if l.startswith("\\item Jude,")]
assert len(jude) == 1, "Jude item not found"

newbib = (
"\\bibliographystyle{plainnat}\n"
"\\bibliography{refs}\n"
"% The reference below is retained verbatim from the hand-built list per task v5-cite-pass 2b.4:\n"
"% no refs.bib entry exists for it (no DOI; arXiv record unretrievable), so it cannot be \\cite'd.\n"
"\\begin{itemize}\n" + jude[0] + "\n\\end{itemize}\n"
)
src = src[:m.start()] + newbib + src[m.end():]

open(P, "w", encoding="utf-8").write(src)
print(f"\nCITATION hunks applied: {len(REPL)} sites, {sum(r[3] for r in REPL)} replacements")
print(f"BIBLIOGRAPHY hunk: 31 \\item removed, 30 keyed -> \\bibliography{{refs}}, 1 (Jude) retained verbatim")
print("md5 new:", hashlib.md5(src.encode()).hexdigest())
