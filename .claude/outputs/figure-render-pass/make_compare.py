"""Before/after sheets: each figure rasterised at EXACTLY its printed width, so the
comparison is at reading size.  300 dpi."""
import os
from PIL import Image, ImageDraw

DPI = 300
PAIRS = [
    ("V2 Fig 1  0.68 linewidth = 3.740 in", "fig1_mo_headtohead.png", 3.740,
     ".claude/scratch/figure-render-pass/figs_before", ".claude/papers/neurreps-variants/v2/figs"),
    ("V2 Fig 2  0.86 linewidth = 4.730 in", "fig2_anchor_cure_laws.png", 4.730,
     ".claude/scratch/figure-render-pass/figs_before", ".claude/papers/neurreps-variants/v2/figs"),
    ("V2 Fig 3  0.60 textwidth = 3.300 in", "fig3_gmor_condensate.png", 3.300,
     ".claude/scratch/figure-render-pass/figs_before", ".claude/papers/neurreps-variants/v2/figs"),
    ("V5 Fig 1  0.60 linewidth = 3.300 in", "fig1_damping_optimum.png", 3.300,
     ".claude/scratch/figure-render-pass/figs_before", ".claude/papers/palm-variant/v5/figs"),
    ("V5 Fig 2  0.74 linewidth = 4.070 in", "fig2_two_instruments.png", 4.070,
     ".claude/scratch/figure-render-pass/figs_before", ".claude/papers/palm-variant/v5/figs"),
    ("V5 Fig C.2  0.80 linewidth = 4.400 in", "figC2_vault_emergent.png", 4.400,
     ".claude/scratch/figure-render-pass/figs_before", ".claude/papers/palm-variant/v5/figs"),
]
OUT = ".claude/outputs/figure-render-pass"
os.makedirs(OUT, exist_ok=True)
for title, name, w_in, dbefore, dafter in PAIRS:
    W = int(round(w_in * DPI))
    ims = []
    for lab, d in (("BEFORE (as printed today)", dbefore), ("AFTER (this pass)", dafter)):
        im = Image.open(os.path.join(d, name)).convert("RGB")
        h = int(round(W * im.size[1] / im.size[0]))
        ims.append((lab, im.resize((W, h), Image.LANCZOS)))
    pad, hdr = 24, 34
    H = sum(hdr + i.size[1] + pad for _, i in ims) + pad
    sheet = Image.new("RGB", (W + 2 * pad, H + hdr), "white")
    dr = ImageDraw.Draw(sheet)
    dr.text((pad, 8), title + "   (rasterised at the printed width, 300 dpi)", fill="black")
    y = hdr + pad
    for lab, im in ims:
        dr.text((pad, y - 18), lab, fill="black")
        sheet.paste(im, (pad, y))
        y += im.size[1] + pad + hdr
    p = os.path.join(OUT, "compare_" + name)
    sheet.save(p, dpi=(DPI, DPI))
    print("wrote", p, sheet.size)
