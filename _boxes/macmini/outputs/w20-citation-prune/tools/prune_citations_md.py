import re,sys
src,bib,dst=sys.argv[1:4]
keys=set(re.findall(r'^@\w+\{([^,]+),',open(bib).read(),re.M))
lines=open(src).read().split('\n'); out=[]; dropped=[]
for l in lines:
    m=re.match(r'\| `([a-z0-9]+)` \|',l)
    if m and m.group(1)!='lecun2022path' and m.group(1) not in keys:
        dropped.append(m.group(1)); continue
    out.append(l)
t='\n'.join(out)
note=("\n> ⭐ **w20 `citation-prune` (2026-09-23): this file is PRUNED to the 55 keys in `references.bib`.** "
      "Rows for the 46 dropped keys (13 cited-for-colour + 33 never cited) were removed; each is listed with its reason in "
      "`~/clu/outputs/w20-citation-prune/CITATION-PRUNE.md`, and the full pre-prune file is `before-20260923T181535Z/CITATIONS.md` there. "
      "Five rows were added (§1h and §5b), each identifier resolved by w20 against Crossref / the arXiv API (`verify/`). "
      "Since w20 `referee3-fixes`, the paper makes **no emptiness claim** (U-1 below is historical); the screen counts are reported as counts.\n")
t=t.replace("\n---\n", note+"\n---\n",1)
h1=("\n### 1h. Systems built on top of units (out of scope by definition; one sentence each) — added by w20 citation-prune\n\n"
"| cite key | title | first author | year | arXiv | DOI | what in our draft it supports |\n|---|---|---|---|---|---|---|\n"
"| `golatkar2023training` | Training Data Protection with Compositional Diffusion Models (compartmentalized diffusion, CDM) | Golatkar | 2023 | 2308.01937 | no DOI | §1.1 systems sentence: composes per-source diffusion models and forgets a source by dropping its model — a system, not held-out factor values within one unit. Venue not verified (arXiv only). |\n"
"| `raissi2019physics` | Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations | Raissi | 2019 | no arXiv | 10.1016/j.jcp.2018.10.045 | §1.1 systems sentence: a physical loss added to an MLP unit; our rival is the plain MLP. J. Comput. Phys. 378. (The two 2017 arXiv preprints 1711.10561/1711.10566 have different titles and are not this paper.) |\n"
"| `greydanus2019hamiltonian` | Hamiltonian Neural Networks | Greydanus | 2019 | 1906.01563 | no DOI | §1.1 systems sentence: physical structure added to an MLP unit. NeurIPS 2019 (arXiv comment). |\n")
t=t.replace("\n---\n\n## §2", h1+"\n---\n\n## §2",1)
h5=("\n### 5b. The shorts' key references, named in the draft's prose (App. `app:shorts`) — added by w20 citation-prune\n\n"
"| cite key | title | first author | year | arXiv | DOI | what in our draft it supports |\n|---|---|---|---|---|---|---|\n"
"| `blelloch2007strongly` | Strongly History-Independent Hashing with Applications | Blelloch | 2007 | no arXiv | 10.1109/FOCS.2007.36 | ⛔ Mandatory attribution (shorts inventory N118 / P5-5): \"the placement rule is the Blelloch--Golovin stable-matching table … the theorems follow from theirs and the fix-up cascade is theirs\". FOCS 2007. |\n"
"| `gellmann1968behavior` | Behavior of Current Divergences under SU3×SU3 | Gell-Mann | 1968 | no arXiv | 10.1103/PhysRev.175.2195 | The GMOR relation named in the \"GMOR on trained checkpoints\" law (μ²F² = δΣ). Physical Review 175. |\n")
t=t.replace("\n---\n\n## §6", h5+"\n---\n\n## §6",1)
assert t.count('### 1h.')==1 and t.count('### 5b.')==1
t=t.replace("## Summary counts (computed from this file's tables, not by hand)",
 "## Summary counts\n**After the w20 prune:** 55 entries (= `references.bib`), 37 arXiv IDs, 31 DOIs (see `CITATIONS-FOR-ZOTERO.md`). The block below is the w17 citation-scout's pre-prune tally, kept for provenance.\n\n### Pre-prune (w17) summary counts (computed from this file's tables, not by hand)")
open(dst,'w').write(t)
rows=set(re.findall(r'^\| `([a-z0-9]+)` \|',t,re.M))-{'lecun2022path'}
print('dropped rows',len(dropped),'| remaining keyed rows',len(rows),'| == bib keys:',rows==keys, sorted(rows^keys))
