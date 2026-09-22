#!/usr/bin/env python3
"""w17 cite-tagger: build references.bib automatically from DOI (Crossref) / arXiv metadata.

Input : ~/clu/outputs/w17-paper/CITATIONS.md  (citation-scout; keys, arXiv IDs, DOIs, venues)
Output: references.bib + build_bib_report.json (per-entry source, checks, marks)

Rules (task file):
  * never fabricate an entry: every author/title/year comes from a fetched record
    (Crossref for DOI entries, the arXiv API for arXiv-only entries, Semantic Scholar only to confirm a
    venue the scout flagged (DBLP is bot-walled here) or for the one entry with neither identifier);
  * entries the scout marked `no DOI` / `no arXiv` are built from what exists and marked
    in a `note` field;
  * lecun2022path is excluded (scout: unverified, do not insert).
"""
import html, json, re, sys, time, unicodedata, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

CIT = Path.home() / "clu/outputs/w17-paper/CITATIONS.md"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "references.bib")
CACHE = OUT.parent / "meta_cache.json"
UA = {"User-Agent": "w17-cite-tagger/1.0 (mailto:info@forgis.com)"}
EXCLUDE = {"lecun2022path"}

# Venue for arXiv-only entries, transcribed from CITATIONS.md's "supports" column
# (scout-verified from the arXiv comment / journal-ref, or the positioning scout).
# None = scout wrote "venue not verified" or gave none -> @misc arXiv preprint.
# ("confirm", ...) = scout asked cite-tagger to confirm -> checked against Semantic Scholar below.
ICLR, ICML, NEURIPS = ("International Conference on Learning Representations",
                       "International Conference on Machine Learning",
                       "Advances in Neural Information Processing Systems")
VENUE = {
    "georgopoulos2020multilinear": ICML, "harkonen2020ganspace": NEURIPS,
    "oldfield2023panda": ICLR, "du2020compositional": NEURIPS, "du2023reduce": ICML,
    "okawa2023compositional": NEURIPS, "ginart2019making": NEURIPS, "xu2023netflix": None,
    "guo2020certified": ICML,
    "izzo2021approximate": "International Conference on Artificial Intelligence and Statistics",
    "kong2022approximate": None, "liu2024machine": None, "fang2026machine": None,
    "fan2024salun": ICLR, "priem2022openalex": None,
    "hafner2019dream": ("confirm", ICLR, "ICLR"), "micheli2023transformers": ICLR,
    "valevski2025diffusion": ICLR, "bruce2024genie": None, "kipf2020contrastive": ICLR,
    "zhao2022toward": ICML, "kim2018disentangling": ("confirm", ICML, "ICML"),
    "schott2021visual": ("confirm", ICLR, "ICLR"), "locatello2019challenging": ICML,
    "liang2025compositional": None, "mahajan2025compositional": ICML,
    "lake2018generalization": ICML, "keysers2020measuring": ICLR,
    "vaswani2017attention": NEURIPS, "nie2021controllable": NEURIPS,
    "borgeaud2021improving": ("confirm", ICML, "ICML"),
    "khandelwal2020generalization": ICLR, "lample2019large": NEURIPS,
    "wu2022memorizing": ICLR, "berges2024memory": None, "krotov2016dense": NEURIPS,
    "ramsauer2021hopfield": ("confirm", ICLR, "ICLR"),
    "gu2020hippo": ("confirm", NEURIPS, "NeurIPS"), "gu2022efficiently": ICLR,
    "gu2024mamba": "Conference on Language Modeling", "chen2018neural": NEURIPS,
    "cohen2021gradient": ICLR, "frankle2019lottery": ICLR, "graves2014neural": None,
    "kingma2015adam": ICLR, "ha2018world": None,
    "kidger2021equinox": "Differentiable Programming Workshop at Neural Information Processing Systems",
}


def get(url, accept=None, tries=4):
    h = dict(UA)
    if accept:
        h["Accept"] = accept
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=40) as r:
                return r.read().decode("utf-8")
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"GET {url}: {err}")


def parse_citations():
    rows = {}
    for line in CIT.read_text().splitlines():
        m = re.match(r"^\| `([a-z0-9]+)` \|(.*)$", line)
        if not m:
            continue
        cells = [c.strip() for c in m.group(2).split("|")]
        key, (title, first, year, arx, doi, notes) = m.group(1), cells[:6]
        if title.startswith("(see"):
            continue
        arx = None if arx.startswith("no arXiv") else arx
        doi = None if doi.startswith("no DOI") else doi
        rows[key] = dict(title=title, first=first, year=year, arxiv=arx, doi=doi, notes=notes)
    return rows


# ---------------------------------------------------------------- text helpers
ACC = {"̀": "`", "́": "'", "̂": "^", "̈": '"', "̃": "~",
       "̧": "c", "̌": "v", "̄": "=", "̆": "u", "̊": "r"}
SPECIAL = {"ø": r"{\o}", "Ø": r"{\O}", "ß": r"{\ss}", "æ": r"{\ae}", "ł": r"{\l}",
           "Ł": r"{\L}", "đ": r"{\dj}", "–": "--", "—": "---", "’": "'", "‘": "`",
           "“": "``", "”": "''", "ı": r"{\i}", " ": "~", "&": r"\&", "%": r"\%",
           "#": r"\#", "_": r"\_", "×": r"$\times$", "…": r"\ldots{}"}


def latexify(s):
    s = html.unescape(s)  # Crossref escapes its JATS/MathML markup
    s = re.sub(r"<tex-math[^>]*>\s*\\\((.*?)\\\)\s*</tex-math>", r"$\1$", s, flags=re.S)
    s = re.sub(r"<mml:math.*?</mml:math>", "", s, flags=re.S)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\s*(\$[^$]+\$)\s*(?=-)", r" \1", s)
    s = re.sub(r"\s+", " ", s).strip()
    out = []
    for ch in s:
        if ch in SPECIAL:
            out.append(SPECIAL[ch]); continue
        d = unicodedata.normalize("NFD", ch)
        if len(d) == 2 and d[1] in ACC:
            base = r"\i" if d[0] == "i" else d[0]
            out.append("{\\%s{%s}}" % (ACC[d[1]], base) if ACC[d[1]].isalpha()
                       else "{\\%s%s}" % (ACC[d[1]], base))
        elif ord(ch) < 128:
            out.append(ch)
        else:
            out.append(ch)  # left as UTF-8; reported below
    return "".join(out)


def protect_title(t):
    # brace words the bst would otherwise lowercase: acronyms, CamelCase, digits, colon-heads
    def w(m):
        tok = m.group(0)
        if re.search(r"[A-Z].*[A-Z]|\d", tok) or (tok[:1].isupper() and len(tok) > 1 and
                                                  any(c.isupper() for c in tok[1:])):
            return "{" + tok + "}"
        return tok
    return re.sub(r"(?<![\\{])\b[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9]\b", w, t)


def norm(s):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", s).lower())


def names_bib(authors):
    return " and ".join(authors)


# ---------------------------------------------------------------- sources
def crossref(doi):
    j = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    authors = []
    for a in j.get("author", []):
        if "family" in a:
            authors.append(latexify(f"{a['family']}, {a.get('given','')}".strip().rstrip(",")))
        elif "name" in a:
            authors.append("{" + latexify(a["name"]) + "}")
    issued = (j.get("issued") or j.get("published") or {}).get("date-parts", [[None]])[0][0]
    return dict(src="crossref", type=j.get("type"), title=latexify((j.get("title") or [""])[0]),
                container=latexify(next((c for c in (j.get("container-title") or [])
                                         if c != "Lecture Notes in Computer Science"),
                                        (j.get("container-title") or [""])[0])),
                series=("Lecture Notes in Computer Science"
                        if "Lecture Notes in Computer Science" in (j.get("container-title") or [])
                        else None),
                volume=j.get("volume"), issue=j.get("issue"), pages=j.get("page"),
                publisher=latexify(j.get("publisher", "")), year=issued, authors=authors)


def arxiv(aid):
    x = get("https://export.arxiv.org/api/query?id_list=" + aid)
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    e = ET.fromstring(x).find("a:entry", ns)
    title = latexify(e.find("a:title", ns).text)
    authors = []
    for a in e.findall("a:author", ns):
        n = a.find("a:name", ns).text.strip()
        parts = n.split()
        authors.append(latexify(f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else n))
    pub = e.find("a:published", ns).text[:4]
    prim = e.find("x:primary_category", ns)
    return dict(src="arxiv", title=title, authors=authors, year=int(pub),
                primary=prim.attrib.get("term") if prim is not None else None)


def s2(path):
    # Semantic Scholar Graph API (DBLP is bot-walled from this box); back off on 429
    for i in range(12):
        try:
            return json.loads(get("https://api.semanticscholar.org/graph/v1/" + path, tries=1))
        except Exception:  # noqa: BLE001
            time.sleep(10 + 5 * i)
    raise RuntimeError("semantic scholar unreachable: " + path)


def dblp_confirm(arxiv_id, token, year):
    """Confirm a venue via Semantic Scholar: venue name or DBLP key must name it, same year."""
    time.sleep(4)
    j = s2(f"paper/arXiv:{arxiv_id}?fields=title,venue,year,externalIds")
    venue, dblp = j.get("venue") or "", (j.get("externalIds") or {}).get("DBLP") or ""
    long = {"ICLR": "learning representations", "ICML": "machine learning",
            "NeurIPS": "neural information processing"}[token]
    conf = {"ICLR": "conf/iclr/", "ICML": "conf/icml/", "NeurIPS": "conf/nips/"}[token]
    ok = (str(j.get("year")) == str(year) and long in venue.lower()) or (
        dblp.startswith(conf) and dblp.endswith(str(year)[2:]))
    how = "S2 venue+year" if ok else None
    # S2 often dates a paper by its arXiv year and keeps a CoRR DBLP key; accept the venue
    # NAME from S2 when its year is the scout's venue year minus one (preprint year).
    if not ok and long in venue.lower() and j.get("year") and int(j["year"]) == int(year) - 1:
        ok, how = True, "S2 venue name; venue year per citation-scout (S2 dates the preprint)"
    return ok, {"venue": venue, "year": j.get("year"), "dblp": dblp, "how": how}


def dblp_record(title):
    time.sleep(4)
    j = s2("paper/search/match?query=" + urllib.parse.quote(title)
           + "&fields=title,venue,year,authors,externalIds")
    for i in j.get("data", []):
        if norm(i.get("title", "")).startswith(norm(title)[:30]):
            return i
    return None


# ---------------------------------------------------------------- build
def main():
    rows = parse_citations()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    report, entries = {}, []
    for key, r in rows.items():
        rep = report.setdefault(key, {"marks": []})
        if key in EXCLUDE:
            rep["status"] = "EXCLUDED (scout: unverified, do not insert)"
            continue
        f = {}
        if r["doi"] and not r["doi"].startswith("10.5281/"):
            m = cache.get("doi:" + r["doi"]) or crossref(r["doi"])
            cache["doi:" + r["doi"]] = m
            rep["source"] = "crossref:" + r["doi"]
            f["author"] = names_bib(m["authors"])
            f["title"] = protect_title(m["title"])
            f["year"] = str(m["year"])
            f["doi"] = r["doi"]
            t = m["type"]
            if t == "journal-article":
                etype = "article"; f["journal"] = m["container"]
                for k in ("volume", "issue", "pages"):
                    if m.get(k):
                        f["number" if k == "issue" else k] = str(m[k]).replace("-", "--")
            elif t in ("proceedings-article", "book-chapter"):
                etype = "inproceedings"; f["booktitle"] = m["container"]
                if m.get("pages"):
                    f["pages"] = m["pages"].replace("-", "--")
                if m.get("publisher"):
                    f["publisher"] = m["publisher"]
                if m.get("series"):
                    f["series"] = m["series"]
            else:
                etype = "misc"; f["howpublished"] = m["container"] or m["publisher"]
                rep["marks"].append(f"crossref type {t}")
            rep["title_match"] = norm(m["title"])[:25] == norm(r["title"])[:25]
            if str(m["year"]) != r["year"]:
                rep["marks"].append(f"year: Crossref issued {m['year']} vs key year {r['year']}")
            if r["arxiv"]:
                f["eprint"] = r["arxiv"]; f["archiveprefix"] = "arXiv"
            else:
                f["note"] = "No arXiv version found by the citation scout"
        elif r["arxiv"]:
            m = cache.get("arxiv:" + r["arxiv"]) or arxiv(r["arxiv"])
            cache["arxiv:" + r["arxiv"]] = m
            rep["source"] = "arxiv:" + r["arxiv"]
            f["author"] = names_bib(m["authors"])
            f["title"] = protect_title(m["title"])
            rep["title_match"] = norm(m["title"])[:25] == norm(r["title"])[:25]
            f["eprint"] = r["arxiv"]; f["archiveprefix"] = "arXiv"
            if m.get("primary"):
                f["primaryclass"] = m["primary"]
            v = VENUE.get(key, "MISSING")
            if isinstance(v, tuple):
                ok, found = cache.get("dblp:" + key) or dblp_confirm(r["arxiv"], v[2], r["year"])
                cache["dblp:" + key] = [ok, found]
                rep["dblp_confirm"] = {"ok": ok, "expected": f"{v[2]} {r['year']}", "s2": found}
                v = v[1] if ok else None
                if not ok:
                    rep["marks"].append("venue NOT confirmed on Semantic Scholar -> arXiv preprint")
            if v == "MISSING":
                raise SystemExit(f"no venue decision for {key}")
            if r["doi"]:  # Zenodo DataCite DOI (Ha & Schmidhuber): kept, resolved by the scout
                f["doi"] = r["doi"]
            if v:
                etype = "inproceedings"; f["booktitle"] = v; f["year"] = r["year"]
                f["note"] = "No DOI resolved by the citation scout"
            else:
                etype = "misc"; f["year"] = str(m["year"])
                f["howpublished"] = f"arXiv preprint arXiv:{r['arxiv']}"
                f["note"] = "Venue not verified; cited as arXiv preprint" + (
                    "" if r["doi"] else "; no DOI resolved by the citation scout")
                rep["marks"].append("arXiv-only (venue not verified)")
        else:
            rep["source"] = "neither arXiv nor DOI"
            if key == "burgess2018shapes":
                readme = cache.get("burgess") or get(
                    "https://raw.githubusercontent.com/google-deepmind/3d-shapes/master/README.md")
                cache["burgess"] = readme
                bm = re.search(r"@misc\{3dshapes18,(.*?)\n\}", readme, re.S)
                if not bm:
                    raise SystemExit("3d-shapes README bibtex not found")
                body = bm.group(1)
                gf = lambda k: re.search(k + r"\s*=\s*\{(.*?)\}\s*(,|$)", body, re.S).group(1)
                etype = "misc"
                f = {"author": gf("author"), "title": "{" + gf("title") + "}",
                     "howpublished": r"\url{https://github.com/google-deepmind/3d-shapes}",
                     "year": gf("year"),
                     "note": "Dataset release; no arXiv, no DOI; metadata from the repository README"}
                rep["source"] = "github README bibtex (google-deepmind/3d-shapes)"
            elif key == "montero2021role":
                i = cache.get("dblp:montero") or dblp_record(r["title"])
                cache["dblp:montero"] = i
                if not i:
                    raise SystemExit("montero not on DBLP")
                names = []
                for a in i["authors"]:
                    n = a["name"].split()
                    names.append(latexify(f"{n[-1]}, {' '.join(n[:-1])}"))
                if "learning representations" not in (i.get("venue") or "").lower():
                    raise SystemExit("montero venue mismatch")
                etype = "inproceedings"
                f = {"author": names_bib(names), "title": protect_title(latexify(i["title"])),
                     "booktitle": ICLR, "year": str(i["year"]),
                     "url": "https://openreview.net/forum?id=qbH974jKUVy",
                     "note": "No arXiv, no DOI; metadata from Semantic Scholar (DBLP key "
                             + (i.get("externalIds") or {}).get("DBLP", "?")
                             + "); OpenReview not fetched (blocked)"}
                rep["source"] = "semanticscholar:" + i.get("paperId", "")
            else:
                raise SystemExit(f"no source for {key}")
            rep["marks"].append("no arXiv, no DOI")
        entries.append((key, etype, f))
        CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
        rep["status"] = "OK"
        rep["type"] = etype
    CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    order = ["author", "title", "journal", "booktitle", "volume", "number", "pages", "series", "publisher",
             "year", "howpublished", "doi", "eprint", "archiveprefix", "primaryclass", "url", "note"]
    with OUT.open("w") as fh:
        fh.write("% references.bib -- w17 cite-tagger. GENERATED by build_bib.py from Crossref (DOI),\n"
                 "% the arXiv API (arXiv-only entries) and Semantic Scholar (venue confirmation / Montero 2021).\n"
                 "% Keys and identifiers: ~/clu/outputs/w17-paper/CITATIONS.md (citation-scout).\n"
                 "% Do not hand-edit; re-run build_bib.py.\n\n")
        for key, etype, f in entries:
            fh.write(f"@{etype}{{{key},\n")
            for k in order:
                if f.get(k):
                    fh.write(f"  {k} = {{{f[k]}}},\n")
            fh.write("}\n\n")
    nonascii = [k for k, _, f in entries if any(ord(c) > 127 for c in "".join(f.values()))]
    (OUT.parent / "build_bib_report.json").write_text(json.dumps(
        {"entries_written": len(entries), "excluded": sorted(EXCLUDE), "non_ascii_left": nonascii,
         "per_key": report}, indent=1, ensure_ascii=False))
    print("entries", len(entries), "non-ascii", nonascii)
    print("title mismatches", [k for k, v in report.items() if v.get("title_match") is False])
    print("marks", {k: v["marks"] for k, v in report.items() if v.get("marks")})


if __name__ == "__main__":
    main()
