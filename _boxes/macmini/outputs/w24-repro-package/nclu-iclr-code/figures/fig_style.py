"""fig style."""

import os

os.environ["SOURCE_DATE_EPOCH"] = "0"
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from nclu.paths import out_root as _out_root

EVIDENCE = str(_out_root())
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3de"
FLOOR = "#6b6a66"
CONTROL = "#b5b4ae"
ARMS = {
    "clu": dict(name="CLU", color="#2a78d6", marker="o"),
    "learned_1x": dict(name="Learned 1× (four-part total)", color="#eb6834", marker="s"),
    "mlp": dict(name="MLP", color="#1baf7a", marker="^"),
    "lstm": dict(name="LSTM", color="#eda100", marker="v"),
    "transformer": dict(name="Transformer", color="#e87ba4", marker="D"),
    "cont_mlp": dict(name="Continuous MLP", color="#008300", marker="P"),
    "table": dict(name="Lookup table (floor)", color=FLOOR, marker="x"),
    "controls": dict(name="landscape controls", color=CONTROL, marker="."),
    "adapter": dict(name="CLU + adapter", color="#4a3aa7", marker="s"),
    "anchor": dict(name="CLU + anchor", color="#e34948", marker="^"),
}
PARTS = [
    ("p1_atom_bytes", "atoms", "#b7d3f2"),
    ("p2_covariance_plus_count_bytes", "covariance + count", "#6fa6e6"),
    ("p3_address_book_bytes", "address book", "#2a78d6"),
    ("p4_coupling_hole_table_bytes", "coupling / hole table", "#0d3f7e"),
]
TEXT_W = 5.5
plt.rcParams.update(
    {
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
        "svg.hashsalt": "figures",
        "path.simplify": False,
    }
)
import re

FIGURE_TEXT_BANNED = re.compile(
    "(?<!n-)\\bCLU\\b|rival|corner|\\bstore\\b|\\bgate\\b|artefact|artifact|pre-?regist|\\bA1\\b|\\bA10\\b|C-512|improved|config|study|\\bhub\\b|\\b\\b|protocol|footprint|oblong|exact|budget|Transf\\.|Cont\\. MLP|∧|10x|\\bpric|\\bbuys?\\b|\\bpaid\\b|\\bpays?\\b|run|landscape_|random_atoms|composition \\(|held-out composition",
    re.IGNORECASE,
)
FIGURE_TEXT_SEEN = []


def check_figure_vocabulary(fig):
    fig.canvas.draw()
    seen = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if not s:
            continue
        m = FIGURE_TEXT_BANNED.search(s)
        assert m is None, f"figure vocabulary: {m.group(0)!r} in rendered text {s!r}"
        seen.append(s)
    FIGURE_TEXT_SEEN.append(sorted(set(seen)))
    return sorted(set(seen))
