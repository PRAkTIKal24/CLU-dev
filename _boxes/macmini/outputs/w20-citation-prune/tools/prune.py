#!/usr/bin/env python3
"""w20 citation-prune: edit cites in both drafts, write pruned references.bib.
Usage: prune.py <src_dir> <dst_dir>   (src = pre-prune snapshot, dst = w17-paper)"""
import re, sys, pathlib
src, dst = map(pathlib.Path, sys.argv[1:3])

DROP = ["nguyen2025survey", "okawa2023compositional", "du2023reduce", "hall1999adding",
        "hafner2019dream", "schrittwieser2020mastering", "micheli2023transformers",
        "bruce2024genie", "alonso2024diffusion", "izzo2021approximate", "wu2022memorizing",
        "berges2024memory", "kipf2020contrastive"]

# (old, new, expected count per file) -- insertions of the five added keys + comment refresh
INSERT = [
    ("(Golatkar, Achille, Swaminathan and Soatto)",
     "(Golatkar, Achille, Swaminathan and Soatto\\nocite{golatkar2023training})", 1),
    ("and Karniadakis, 2019)", "and Karniadakis, 2019\\nocite{raissi2019physics})", 1),
    ("Dzamba and Yosinski, 2019)", "Dzamba and Yosinski, 2019\\nocite{greydanus2019hamiltonian})", 1),
    ("The placement rule is the Blelloch--Golovin stable-matching table",
     "The placement rule is the Blelloch--Golovin stable-matching table~\\citep{blelloch2007strongly}", 1),
    ("GMOR on trained checkpoints.}", "GMOR~\\citep{gellmann1968behavior} on trained checkpoints.}", 1),
    ('% cite-tagger: bib entry needed -- Golatkar, Achille, Swaminathan, Soatto, "Training Data Protection with Compositional Diffusion Models", arXiv:2308.01937 (not in references.bib)',
     '% cite-tagger: bib entry added by w20 citation-prune -- golatkar2023training (Golatkar, Achille, Swaminathan, Soatto, "Training Data Protection with Compositional Diffusion Models", arXiv:2308.01937)', 1),
    ('% cite-tagger: bib entries needed -- Raissi, Perdikaris, Karniadakis, "Physics-informed neural networks", J. Comput. Phys. (2019); Greydanus, Dzamba, Yosinski, "Hamiltonian Neural Networks", NeurIPS 2019 (neither in references.bib)',
     '% cite-tagger: bib entries added by w20 citation-prune -- raissi2019physics (Raissi, Perdikaris, Karniadakis, "Physics-informed neural networks", J. Comput. Phys. 378, 2019); greydanus2019hamiltonian (Greydanus, Dzamba, Yosinski, "Hamiltonian Neural Networks", NeurIPS 2019)', 1),
]

CITE = re.compile(r'(~?)\\(nocite|citep|citet|cite)\{([^}]*)\}')

def drop_keys(text):
    removed = []
    def rep(m):
        tilde, cmd, keys = m.groups()
        ks = [k.strip() for k in keys.split(",")]
        kept = [k for k in ks if k not in DROP]
        removed.extend(k for k in ks if k in DROP)
        if not kept:
            return ""
        return f"{tilde}\\{cmd}{{{','.join(kept)}}}"
    return CITE.sub(rep, text), removed

for name in ["DRAFT-v2.tex", "DRAFT-v2-style.tex"]:
    t = (src / name).read_text()
    t, removed = drop_keys(t)
    assert sorted(removed) == sorted(DROP), (name, sorted(set(DROP) - set(removed)), removed)
    for old, new, n in INSERT:
        c = t.count(old)
        assert c == n, (name, old, c)
        t = t.replace(old, new)
    (dst / name).write_text(t)
    print(name, "edited; removed", len(removed), "cite keys; inserted", len(INSERT) - 2, "citations")

# ---- bib ----
bib = (src / "references.bib").read_text()
head, *entries = re.split(r'\n(?=@)', bib)
ent = {re.match(r'@\w+\{([^,]+),', e).group(1): e.rstrip() for e in entries}
assert len(ent) == 96, len(ent)
cited = set()
for name in ["DRAFT-v2.tex", "DRAFT-v2-style.tex"]:
    for m in CITE.finditer((dst / name).read_text()):
        cited.update(k.strip() for k in m.group(3).split(",") if k.strip())
NEW = (src.parent / "tools" / "new_entries.bib") if False else pathlib.Path(__file__).with_name("new_entries.bib")
newtxt = NEW.read_text()
newent = {re.match(r'@\w+\{([^,]+),', e).group(1): e.strip() for e in re.split(r'\n(?=@)', newtxt.strip())}
keep = [k for k in ent if k in cited]
missing = cited - set(ent) - set(newent)
assert not missing, missing
out = ["%% references.bib -- w17 cite-tagger, PRUNED by w20 citation-prune (96 -> %d entries).\n"
       "%% Kept entries are byte-identical to the w17 cite-tagger's (build_bib.py: Crossref / arXiv API /\n"
       "%% Semantic Scholar). The last %d entries were added by w20 citation-prune from Crossref and the\n"
       "%% arXiv API (verify/ in outputs/w20-citation-prune/). Every entry is cited by both drafts.\n"
       "%% Keys and identifiers: CITATIONS.md; prune record: outputs/w20-citation-prune/CITATION-PRUNE.md.\n"
       % (len(keep) + len(newent), len(newent))]
for k in keep:
    out.append(ent[k] + "\n")
for k, e in newent.items():
    assert k in cited, k
    out.append(e + "\n")
(dst / "references.bib").write_text("\n".join(out))
print("bib:", len(keep), "kept +", len(newent), "new =", len(keep) + len(newent))
print("KEPT", " ".join(keep + list(newent)))
print("DROPPED_UNCITED", " ".join(k for k in ent if k not in cited and k not in DROP))
