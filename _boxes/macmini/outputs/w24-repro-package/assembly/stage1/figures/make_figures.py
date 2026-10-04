"""w17/plots -- the paper's figures, built from w<=14 evidence artefacts only.

Every plotted value is read from a file under EVIDENCE (read-only). Nothing is typed in by
hand, rounded or recalled; where a figure shows a mean it is either the artefact's own mean or
is checked against it. Provenance for each figure is written to <out>/PROVENANCE.json.

AMENDMENT 1 (Head, 2026-09-22): no w15 or w16 artefact is read. Figures that wave 18 must
regenerate carry a PLACEHOLDER tag and are listed in PROVENANCE.json.

Determinism: Agg/PDF backends pinned, bundled DejaVu Sans, no timestamps in PDF/PNG metadata,
inputs read in sorted order, no randomness. Build twice and compare shasums (see FINDINGS.md).

usage: python w17_plots/make_figures.py --out <dir>
"""
import os
os.environ["SOURCE_DATE_EPOCH"] = "0"
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from chlu.paths import out_root as _out_root
EVIDENCE = str(_out_root())
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3de"
FLOOR = "#6b6a66"
CONTROL = "#b5b4ae"
ARMS = {
    "clu":         dict(name="CLU", color="#2a78d6", marker="o"),
    "learned_1x":  dict(name="Learned 1× (four-part total)", color="#eb6834", marker="s"),
    "mlp":         dict(name="MLP", color="#1baf7a", marker="^"),
    "lstm":        dict(name="LSTM", color="#eda100", marker="v"),
    "transformer": dict(name="Transformer", color="#e87ba4", marker="D"),
    "cont_mlp":    dict(name="Continuous MLP", color="#008300", marker="P"),
    "table":       dict(name="Lookup table (floor)", color=FLOOR, marker="x"),
    "controls":    dict(name="Charter-C controls", color=CONTROL, marker="."),
    # encoder-adapt configurations of the CLU (slots 7, 8)
    "adapter":     dict(name="CLU + adapter", color="#4a3aa7", marker="s"),
    "anchor":      dict(name="CLU + anchor", color="#e34948", marker="^"),
}
PARTS = [
    ("p1_atom_bytes", "atoms", "#b7d3f2"),
    ("p2_covariance_plus_count_bytes", "covariance + count", "#6fa6e6"),
    ("p3_address_book_bytes", "address book", "#2a78d6"),
    ("p4_coupling_hole_table_bytes", "coupling / hole table", "#0d3f7e"),
]
TEXT_W = 5.5  # ICLR text width, inches
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "legend.frameon": False,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "pdf.compression": 6,
    "svg.hashsalt": "w17-plots",
    "path.simplify": False,
})
import re  # noqa: E402
PW1_BANNED = re.compile(
    r"(?<!n-)\bCLU\b|rival|corner|\bstore\b|\bgate\b|artefact|artifact|pre-?regist|\bA1\b|"
    r"\bA10\b|C-512|improved|config|spoke|\bhub\b|\bwave\b|charter|ledger|oblong|exact|"
    r"budget|Transf\.|Cont\. MLP|∧|10x|\bpric|\bbuys?\b|\bpaid\b|\bpays?\b|scored run|"
    r"landscape_|random_atoms|composition \(|held-out composition", re.IGNORECASE)
PW1_TEXT = []  # rendered strings per checked figure, in call order (read by build_shapes_only)
def check_vocab_pw1(fig):
    """Assert every rendered string obeys the PW-1 vocabulary; returns the strings checked."""
    fig.canvas.draw()  # tick labels are only set on draw
    seen = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if not s:
            continue
        m = PW1_BANNED.search(s)
        assert m is None, f"PW-1 vocabulary: {m.group(0)!r} in rendered text {s!r}"
        seen.append(s)
    PW1_TEXT.append(sorted(set(seen)))
    return sorted(set(seen))


