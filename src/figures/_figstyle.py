"""One style for the main figures of the manuscript (Figures 1, 3, 4, 5 and 6; the overview figure is drawn by hand).

Every main figure is drawn at its printed size: full text width (178 mm, the journal's maximum for a double-column figure), so that 7 pt in the notebook is 7 pt on
the page. Use it in a notebook's setup cell:

    import _figstyle as fs
    fs.apply()
    fig, axes = plt.subplots(1, 3, figsize=fs.size(rows=1), layout="constrained")
    fs.panel(ax, "a", "Short panel title")
    ...
    fs.save(fig, out_path("fig3"))          # writes fig3.png (600 dpi) and fig3.pdf at exactly that size

Conventions:
  - size       width 178 mm (NARGAB limit); height from fs.size(rows=...) or a fixed value in inches; never bbox_inches="tight"
  - text       7 pt; ticks and legends 6.5 pt; value labels 6 pt (all kept, none overlapping); panel letter 9 pt bold
  - titles     every panel has a short title (2-5 words, sentence case), left-aligned next to its letter
  - axes       sentence case labels with units in brackets; no caption detail in an axis label
  - frames     left and bottom spines only; light horizontal grid on value axes, none on ROC panels
  - names      "Data-driven", "ATAC-seq", "RNA-seq", "ichorCNA"; "AUROC" (never "AUC"); "Cristiano et al.", "Jiang et al.";
               "BRCA/CRC versus healthy", "Pan-cancer versus healthy"
  - colors     METHOD for methods, TASK for the two tasks, FEATURE for the five features (unchanged from before)
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

WIDTH = 178 / 25.4                     # inches, full text width (NARGAB: at most 178 mm)
ROW = 2.35                             # inches per row of panels
FONT, SMALL, VALUE, LETTER = 7, 6.5, 6, 9

METHOD = {"Data-driven": "#1f78b4", "ATAC-seq": "#e6ab02", "RNA-seq": "#d1495b", "ichorCNA": "#6a3d9a"}
METHOD_LIGHT = {"Data-driven": "#8fbcdb"}          # the same method in a second condition (e.g. trained within Jiang et al.)
TASK = {"BRCA/CRC versus healthy": "#0072B2", "Pan-cancer versus healthy": "#E69F00"}
FEATURE = {"FSLR": "#2e2585", "FSLRC": "#337538", "GD": "#5da899", "RCOV": "#dccd7d", "MWH": "#9f4a96"}
NEUTRAL, GRID, CHANCE = "#4d4d4d", "#e3e3e3", "#555555"   # chance lines darker


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT, "axes.titlesize": FONT + 0.5, "axes.labelsize": FONT,
        "xtick.labelsize": SMALL, "ytick.labelsize": SMALL, "legend.fontsize": SMALL, "legend.title_fontsize": SMALL,
        "axes.titlelocation": "left", "axes.titleweight": "normal", "axes.titlepad": 4,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.1, "lines.markersize": 3.5, "patch.linewidth": 0.4,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.3, "legend.labelspacing": 0.3,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5,
        "figure.dpi": 150, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
    })


def size(rows=1, height=None):
    """(width, height) in inches for a full-width figure with `rows` rows of panels."""
    return (WIDTH, height if height is not None else ROW * rows)


def panel(ax, letter, title=None):
    """Short title at the top left of the axes; the bold letter goes to the outer left edge of the panel (its tick
    labels and axis label included), on the baseline of the title. The letter is placed when the figure is saved
    with save(), after the layout is final; before that it sits provisionally 22 pt left of the axes."""
    ax.set_title(title or "", loc="left", pad=4)
    t = ax.text(-0.02, 1.0, letter, transform=ax.transAxes + mpl.transforms.ScaledTranslation(-22 / 72, 4 / 72, ax.figure.dpi_scale_trans),
                fontsize=LETTER, fontweight="bold", va="bottom", ha="left")
    root = ax.get_figure(root=True)
    if not hasattr(root, "_fs_letters"):
        root._fs_letters = []
    root._fs_letters.append((ax, t))


def _place_letters(fig):
    """Move every panel letter to the left edge of its panel's tight box, on the title's baseline."""
    letters = getattr(fig, "_fs_letters", [])
    if not letters:
        return
    fig.canvas.draw()                       # final layout (constrained layout runs here)
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, t in letters:
        width = t.get_window_extent(r).width     # measured while visible (a hidden text has no extent)
        t.set_visible(False)                     # the letter is an axes child: keep it out of the tight box
        box = ax.get_tightbbox(r)
        title = ax._left_title.get_window_extent(r)   # the left-aligned panel title (ax.title is the empty centre one)
        # left edge of the panel, but never on top of the title (axes without y tick labels: Venn, shared y)
        gap = 5 / 72 * fig.dpi              # 5 pt between letter and title
        left = min(box.x0, title.x0 - width - gap) if ax.get_title(loc="left") else box.x0
        x, y = inv.transform((max(left, 2.0), title.y0))
        t.set_transform(fig.transFigure)
        t.set_position((x, y))
        t.set_visible(True)


def value_grid(ax, axis="y"):
    """Light grid on the value axis only, behind the data."""
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, stem):
    """stem without extension; writes <stem>.png (600 dpi) and <stem>.pdf at the figure's own size (no tight crop)."""
    stem = os.path.splitext(stem)[0]
    _place_letters(fig)
    fig.set_layout_engine("none")           # keep the layout the letters were placed for
    for ext in (".png", ".pdf"):
        for attempt in range(4):           # the network drive sometimes refuses a write for a moment
            try:
                fig.savefig(stem + ext)
                break
            except OSError:
                if attempt == 3:
                    raise
    w, h = fig.get_size_inches()
    print(f"saved {stem}.png/.pdf at {w * 25.4:.0f} x {h * 25.4:.0f} mm")
