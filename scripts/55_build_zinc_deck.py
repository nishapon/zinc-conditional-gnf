#!/usr/bin/env python3
"""Build the ZINC continuation section as an editable .pptx.

Styling copied from the master deck: serif display title in dark plum, small-caps
section label, rose accent, pale pink panels, a bottom rule with one bold takeaway, and
a page number bottom right. 8 main slides plus 4 appendix.

Every number is taken from the cluster artefacts (qed_spread.csv, chemistry_by_size.csv,
showcase.csv, architecture.csv). Nothing is invented.
"""

from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_TICK_MARK
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

OUT = Path("/projects/sandbox/ZINC_section_for_master_deck.pptx")

# ---- palette, matched to the master deck -------------------------------------
PLUM = RGBColor(0x3D, 0x1F, 0x2D)        # display title
LABEL = RGBColor(0x7C, 0x6E, 0x9A)       # "06 · ZINC RESULTS"
ROSE = RGBColor(0xA7, 0x36, 0x4E)        # accent, takeaway, big numbers
ROSE_LT = RGBColor(0xC9, 0x6F, 0x84)
PINK_HDR = RGBColor(0xF6, 0xDF, 0xE6)    # table header fill
PINK_BG = RGBColor(0xFC, 0xF4, 0xF7)     # panel fill
BODY = RGBColor(0x3C, 0x3C, 0x42)
GREY = RGBColor(0x6E, 0x6E, 0x76)
PURPLE = RGBColor(0x59, 0x4A, 0x7A)      # unconditional series
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RULE = RGBColor(0xA7, 0x36, 0x4E)

SERIF = "Georgia"
SANS = "Calibri"

W, H = Inches(13.333), Inches(7.5)
L = Inches(0.62)                          # left margin
RGT = Inches(12.71)                       # right edge

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]

PAGE = [0]


def tb(slide, x, y, w, h, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(x, y, w, h)
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = anchor
    frame.paragraphs[0].alignment = align
    return frame


def put(frame, text, size, color, bold=False, font=SANS, space_after=0,
        align=None, first=False, line=None):
    para = frame.paragraphs[0] if first else frame.add_paragraph()
    if align is not None:
        para.alignment = align
    para.space_after = Pt(space_after)
    if line:
        para.line_spacing = line
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    return para


def chrome(slide, label, title, subtitle=None, title_size=34):
    frame = tb(slide, L, Inches(0.34), Inches(8.0), Inches(0.26))
    put(frame, label, 10.5, LABEL, bold=True, first=True)
    frame = tb(slide, L, Inches(0.62), Inches(11.9), Inches(0.72))
    put(frame, title, title_size, PLUM, bold=True, font=SERIF, first=True)
    if subtitle:
        frame = tb(slide, L, Inches(1.38), Inches(11.6), Inches(0.3))
        put(frame, subtitle, 13, GREY, first=True)


def takeaway(slide, text, y=Inches(6.62)):
    line = slide.shapes.add_connector(1, L, y, RGT, y)
    line.line.color.rgb = RULE
    line.line.width = Pt(1.5)
    frame = tb(slide, L, y + Inches(0.10), Inches(11.9), Inches(0.44))
    put(frame, text, 14.5, ROSE, bold=True, first=True)


def page(slide):
    PAGE[0] += 1
    frame = tb(slide, Inches(12.25), Inches(7.08), Inches(0.46), Inches(0.24),
               align=PP_ALIGN.RIGHT)
    put(frame, "%02d" % PAGE[0], 10, GREY, first=True)


def panel(slide, x, y, w, h, fill=PINK_BG, line=None):
    shape = slide.shapes.add_shape(5, x, y, w, h)       # rounded rectangle
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    shape.shadow.inherit = False
    shape.adjustments[0] = 0.04
    return shape


def table(slide, x, y, w, rows, widths=None, header_size=11, body_size=11,
          row_h=Inches(0.33)):
    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, x, y, w, row_h * n_rows)
    tbl = shape.table
    if widths:
        total = sum(widths)
        for index, share in enumerate(widths):
            tbl.columns[index].width = Emu(int(w * share / total))
    for r, row in enumerate(rows):
        tbl.rows[r].height = row_h
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = (PINK_HDR if r == 0
                                        else (PINK_BG if r % 2 else WHITE))
            frame = cell.text_frame
            frame.word_wrap = True
            para = frame.paragraphs[0]
            para.alignment = PP_ALIGN.RIGHT if (c and r) else PP_ALIGN.LEFT
            run = para.add_run()
            run.text = str(value)
            run.font.size = Pt(header_size if r == 0 else body_size)
            run.font.bold = (r == 0)
            run.font.color.rgb = PLUM if r == 0 else BODY
            run.font.name = SANS
    return tbl


def metrics(slide, y, items, width=Inches(11.9)):
    """Big accent numbers with captions, as in the master deck."""
    gap = Inches(0.24)
    each = (width - gap * (len(items) - 1)) / len(items)
    for index, (big, caption) in enumerate(items):
        x = L + index * (each + gap)
        line = slide.shapes.add_connector(1, x, y, x + each, y)
        line.line.color.rgb = RULE
        line.line.width = Pt(1.25)
        frame = tb(slide, x, y + Inches(0.07), each, Inches(0.46))
        put(frame, big, 27, ROSE, bold=False, font=SERIF, first=True)
        frame = tb(slide, x, y + Inches(0.58), each, Inches(0.52))
        put(frame, caption, 10.5, GREY, first=True, line=1.05)


def picture_slot(slide, x, y, w, h, caption, detail=None):
    """Labelled slot for a molecule grid PNG from deck_assets/figures/."""
    shape = panel(slide, x, y, w, h, fill=WHITE, line=RGBColor(0xE3, 0xC9, 0xD3))
    frame = tb(slide, x + Inches(0.12), y + Inches(0.10), w - Inches(0.24),
               Inches(0.26))
    put(frame, caption, 11, ROSE, bold=True, first=True)
    frame = tb(slide, x + Inches(0.12), y + h / 2 - Inches(0.34),
               w - Inches(0.24), Inches(0.7), align=PP_ALIGN.CENTER,
               anchor=MSO_ANCHOR.MIDDLE)
    put(frame, "[ drop in: %s ]" % detail, 10, RGBColor(0xB9, 0xA6, 0xAE),
        first=True, align=PP_ALIGN.CENTER)
    return shape


def smiles_list(slide, x, y, w, rows, title):
    frame = tb(slide, x, y, w, Inches(0.24))
    put(frame, title, 10.5, ROSE, bold=True, first=True)
    frame = tb(slide, x, y + Inches(0.26), w, Inches(1.3))
    for index, (smi, note) in enumerate(rows):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.space_after = Pt(2)
        run = para.add_run()
        run.text = smi
        run.font.size = Pt(8.5)
        run.font.name = "Consolas"
        run.font.color.rgb = BODY
        run = para.add_run()
        run.text = "   " + note
        run.font.size = Pt(8.5)
        run.font.color.rgb = ROSE
        run.font.name = SANS


def style_chart(chart, font=9.5, legend=True):
    chart.font.size = Pt(font)
    chart.font.name = SANS
    chart.font.color.rgb = BODY
    chart.has_title = False
    if legend:
        chart.has_legend = True
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(font - 0.5)
    else:
        chart.has_legend = False


# =============================================================== slide 1 =====
s = prs.slides.add_slide(BLANK)
chrome(s, "06 · ZINC RESULTS", "From QM9 to ZINC250k",
       "Same conditional-flow recipe, a harder dataset and a different property")
rows = [["", "QM9 (earlier section)", "ZINC250k (this section)"],
        ["Molecule size", "up to 9 heavy atoms", "up to 38 heavy atoms"],
        ["Atom vocabulary", "C, N, O, F", "17 (element, formal charge) types"],
        ["Target property", "a physical quantity", "QED, an 8-component composite"],
        ["Node embedding", "--", "128-D, from a frozen autoencoder"],
        ["Flow depth", "--", "12 blocks, 36 coupling layers"],
        ["Assembly", "--", "degree-constrained program over bond logits"]]
table(s, L, Inches(1.86), Inches(11.9), rows, widths=[3.0, 3.2, 4.6],
      row_h=Inches(0.42))
takeaway(s, "Bigger molecules, a richer atom vocabulary, and a property the decoder "
            "cannot be checked against for free.")
page(s)

# =============================================================== slide 2 =====
s = prs.slides.add_slide(BLANK)
chrome(s, "06 · ZINC RESULTS", "One architecture, two condition vectors",
       "The two ZINC models differ only in what the condition carries")

for index, (title, sub, bullets, fill) in enumerate([
    ("Unconditional", "condition_features = [ ]",
     ["No condition embedding",
      "FiLM generators inactive",
      "Prior  N(mu, sigma^2)  fixed per dimension",
      "Atom count still supplied structurally"], PINK_BG),
    ("N + QED  (ours)", "condition_features = ['n', 'qed']",
     ["log(N+1) and standardised QED -> 64-D c",
      "c drives FiLM in all 36 coupling layers",
      "Prior becomes  N(mu(c), sigma(c)^2)",
      "+ QED contrastive objective (next slide)"], PINK_HDR)]):
    x = L + index * Inches(6.07)
    panel(s, x, Inches(1.82), Inches(5.76), Inches(3.42), fill=fill)
    frame = tb(s, x + Inches(0.26), Inches(1.98), Inches(5.3), Inches(0.34))
    put(frame, title, 19, PLUM, bold=True, font=SERIF, first=True)
    frame = tb(s, x + Inches(0.26), Inches(2.36), Inches(5.3), Inches(0.24))
    put(frame, sub, 10.5, ROSE, bold=True, font="Consolas", first=True)
    frame = tb(s, x + Inches(0.26), Inches(2.70), Inches(5.3), Inches(2.4))
    for b_index, text in enumerate(bullets):
        put(frame, "•  " + text, 12, BODY, first=(b_index == 0), space_after=7,
            line=1.08)

frame = tb(s, L, Inches(5.42), Inches(11.9), Inches(0.26))
put(frame, "SHARED BACKBONE, BOTH MODELS", 10.5, LABEL, bold=True, first=True)
rows = [["12 blocks", "ActNorm -> invertible 1x1 -> 2 affine couplings -> "
         "1 node-autoregressive coupling"],
        ["Affine split", "64 / 64 of the 128 channels   ·   "
         "node-autoregressive transforms all 128"],
        ["Scale", "5,178,562 flow parameters   ·   frozen decoder 306,212   ·   "
         "50,000 steps, seed 3"]]
table(s, L, Inches(5.70), Inches(11.9), [["", ""]] + rows,
      widths=[2.0, 9.9], row_h=Inches(0.27), body_size=10.5)
s.shapes[-1].table.rows[0].height = Inches(0.01)
takeaway(s, "A matched pair: identical backbone, identical decoder, identical budget.",
         y=Inches(6.80))
page(s)

# =============================================================== slide 3 =====
s = prs.slides.add_slide(BLANK)
chrome(s, "06 · ZINC RESULTS", "The ZINC modification: a QED contrastive term",
       "Likelihood alone lets the flow ignore QED. This makes ignoring it expensive.")

steps = [("01", "Score the truth",
          "Each molecule is scored under its OWN QED value."),
         ("02", "Score two decoys",
          "The same molecule is re-scored under QED offset by -1 and +1 standard "
          "deviations."),
         ("03", "Push them apart",
          "A softmax over the three scores rewards the true condition. Weight 1.0, "
          "ramped in over steps 2,000 to 4,000.")]
for index, (number, head, text) in enumerate(steps):
    x = L + index * Inches(4.03)
    panel(s, x, Inches(1.82), Inches(3.78), Inches(2.08))
    frame = tb(s, x + Inches(0.24), Inches(1.94), Inches(3.3), Inches(0.4))
    put(frame, number, 22, ROSE, font=SERIF, first=True)
    frame = tb(s, x + Inches(0.24), Inches(2.36), Inches(3.3), Inches(0.26))
    put(frame, head, 13, PLUM, bold=True, first=True)
    frame = tb(s, x + Inches(0.24), Inches(2.66), Inches(3.3), Inches(1.1))
    put(frame, text, 11, BODY, first=True, line=1.08)

rows = [["Setting", "Value", "Why"],
        ["qed_contrast_weight", "1.0", "equal footing with the likelihood term"],
        ["qed_contrast_offsets", "0.0, -1.0, +1.0",
         "decoys one standard deviation away"],
        ["min separation", "0.25", "a decoy must be far enough to be a real decoy"],
        ["warmup / ramp", "2,000 / 2,000 steps",
         "the flow learns the density first, control second"],
        ["subbatch", "32 molecules per step", "cost stays close to plain training"]]
table(s, L, Inches(4.12), Inches(11.9), rows, widths=[2.9, 2.6, 6.4],
      row_h=Inches(0.37), body_size=11)
takeaway(s, "The condition is not just wired in; the objective makes the likelihood "
            "depend on its value.", y=Inches(6.74))
page(s)

# =============================================================== slide 4 =====
s = prs.slides.add_slide(BLANK)
chrome(s, "06 · ZINC RESULTS", "Does the model follow the requested QED?",
       "Achieved mean QED against the request, T = 0.85, 10,000 attempts per point")

data = CategoryChartData()
data.categories = ["0.4", "0.6", "0.8"]
data.add_series("uncond. N=15", (0.5626, 0.5634, 0.5638))
data.add_series("N+QED N=15", (0.5143, 0.5383, 0.5930))
data.add_series("uncond. N=23", (0.4922, 0.4896, 0.4886))
data.add_series("N+QED N=23", (0.4690, 0.4884, 0.5232))
data.add_series("uncond. N=30", (0.3390, 0.3408, 0.3399))
data.add_series("N+QED N=30", (0.3421, 0.3669, 0.3998))
graphic = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, L, Inches(1.80),
                             Inches(7.26), Inches(4.62), data)
chart = graphic.chart
style_chart(chart)
for index, series in enumerate(chart.plots[0].series):
    colour = PURPLE if index % 2 == 0 else ROSE
    series.format.line.color.rgb = colour
    series.format.line.width = Pt(2.0 if index % 2 else 1.5)
    if index % 2 == 0:
        series.format.line.dash_style = 4
axis = chart.value_axis
axis.minimum_scale, axis.maximum_scale = 0.30, 0.65
axis.has_major_gridlines = True
axis.tick_labels.font.size = Pt(9)
chart.category_axis.tick_labels.font.size = Pt(9)
chart.category_axis.major_tick_mark = XL_TICK_MARK.NONE

panel(s, Inches(8.18), Inches(1.80), Inches(4.53), Inches(4.62))
frame = tb(s, Inches(8.44), Inches(1.96), Inches(4.0), Inches(0.3))
put(frame, "RESPONSE SPAN  (0.4 -> 0.8 request)", 10.5, LABEL, bold=True, first=True)
rows = [["Size", "Uncond.", "N+QED"],
        ["N = 15", "0.001", "0.079"],
        ["N = 23", "0.004", "0.054"],
        ["N = 30", "0.002", "0.058"]]
table(s, Inches(8.44), Inches(2.32), Inches(4.0), rows, widths=[1.6, 1.2, 1.2],
      row_h=Inches(0.34), body_size=11.5)
frame = tb(s, Inches(8.44), Inches(3.84), Inches(4.0), Inches(2.3))
put(frame, "Dashed purple lines are flat: the unconditional model produces the same "
           "QED whatever is asked for.", 11.5, BODY, first=True, line=1.12,
    space_after=9)
put(frame, "Solid rose lines rise monotonically at every size.", 11.5, PLUM,
    bold=True, line=1.12, space_after=9)
put(frame, "Response is 15x to 66x larger than the unconditional model's drift.",
    11.5, ROSE, bold=True, line=1.12)
takeaway(s, "Only the conditioned model responds to the request, and at T = 0.85 it "
            "responds monotonically at all three sizes.")
page(s)

# =============================================================== slide 5 =====
s = prs.slides.add_slide(BLANK)
chrome(s, "06 · ZINC RESULTS", "More attempts land on target",
       "Valid molecule within +/-0.1 QED of the request, out of ALL attempts "
       "(T = 0.20, 10,000 per cell)")

data = CategoryChartData()
data.categories = ["N15\n0.4", "N15\n0.6", "N15\n0.8", "N23\n0.4", "N23\n0.6",
                   "N23\n0.8", "N30\n0.4", "N30\n0.6", "N30\n0.8"]
data.add_series("Unconditional", (19.5, 52.4, 24.1, 26.1, 31.5, 27.4, 33.8, 29.7, 7.5))
data.add_series("N + QED", (20.5, 51.7, 36.3, 27.4, 35.3, 28.6, 38.2, 32.6, 15.3))
graphic = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, L, Inches(1.92),
                             Inches(8.2), Inches(4.4), data)
chart = graphic.chart
style_chart(chart)
chart.plots[0].gap_width = 60
chart.plots[0].series[0].format.fill.solid()
chart.plots[0].series[0].format.fill.fore_color.rgb = PURPLE
chart.plots[0].series[1].format.fill.solid()
chart.plots[0].series[1].format.fill.fore_color.rgb = ROSE
axis = chart.value_axis
axis.minimum_scale, axis.maximum_scale = 0, 60
axis.tick_labels.font.size = Pt(9)
axis.has_major_gridlines = True
chart.category_axis.tick_labels.font.size = Pt(9)

panel(s, Inches(9.02), Inches(1.92), Inches(3.69), Inches(4.4))
frame = tb(s, Inches(9.26), Inches(2.10), Inches(3.2), Inches(3.9))
put(frame, "2.0x", 30, ROSE, font=SERIF, first=True)
put(frame, "hit yield at the hardest cell: N = 30, QED = 0.8, 7.5% -> 15.3%",
    11.5, GREY, space_after=16, line=1.1)
put(frame, "+12.2", 30, ROSE, font=SERIF)
put(frame, "percentage points at N = 15, QED = 0.8  (24.1% -> 36.3%)", 11.5, GREY,
    space_after=16, line=1.1)
put(frame, "8 / 9", 30, ROSE, font=SERIF)
put(frame, "grid cells improve; the gain concentrates where the request is far from "
           "what the data offers", 11.5, GREY, line=1.1)
takeaway(s, "The further the request sits from the model's default output, the more "
            "conditioning buys.")
page(s)

# =============================================================== slide 6 =====
s = prs.slides.add_slide(BLANK)
chrome(s, "06 · ZINC RESULTS", "Same request, different chemistry",
       "Nearest-to-target molecules, request N = 30 and QED = 0.8, T = 0.20")
picture_slot(s, L, Inches(1.82), Inches(5.76), Inches(3.30),
             "UNCONDITIONAL", "figures/grid_T0.20_baseline_N30_QED0.8.png")
picture_slot(s, L + Inches(6.07), Inches(1.82), Inches(5.76), Inches(3.30),
             "N + QED CONTRASTIVE", "figures/grid_T0.20_conditional_N30_QED0.8.png")
smiles_list(s, L, Inches(5.22), Inches(5.76), [
    ("CCC1C2CC3=C4C(=C3)NC3N=C4N=C(C)COC(...)C3C1C2", "QED 0.80 · 12 rings"),
    ("CC=Cc1ccccc1CC(C)C(C)C(=O)n1c2nnc1C1NC3C(N2)...", "QED 0.80 ·  7 rings"),
    ("N#CCC(=O)CNCC1=NC2c3ccc2n3C(N=C2CC2C2=CC=...)=N1", "QED 0.80 ·  7 rings")],
    "FUSED POLYCYCLIC CAGES")
smiles_list(s, L + Inches(6.07), Inches(5.22), Inches(5.76), [
    ("CCCOc1ccccc1-c1ccccc1CC(=O)N1C[NH2+]CCC2CCN(C2)C1", "QED 0.80 · 2 aromatic"),
    ("CC(c1ccccc1)N(CN1CCC(=O)CCCN(C)CC1)C(=O)NC1CCCC1", "QED 0.80 · urea"),
    ("CN1CCOCCC1COCc1ccccc1CC1Cc2ccccc2CNC1=O", "QED 0.80 · 2 aromatic")],
    "BIPHENYL, AMIDE, UREA, MORPHOLINE")
takeaway(s, "At the same QED and the same size, the conditioned model reaches for "
            "recognisable medicinal-chemistry motifs.", y=Inches(6.86))
page(s)

# =============================================================== slide 7 =====
s = prs.slides.add_slide(BLANK)
chrome(s, "06 · ZINC RESULTS", "Sampling temperature changes the regime",
       "Same checkpoints, same requests, same budget; only the prior width differs")
picture_slot(s, L, Inches(1.82), Inches(5.76), Inches(2.72),
             "T = 0.20   ·   N + QED   ·   request QED 0.8",
             "figures/grid_T0.20_conditional_N15_QED0.8.png")
picture_slot(s, L + Inches(6.07), Inches(1.82), Inches(5.76), Inches(2.72),
             "T = 0.85   ·   N + QED   ·   request QED 0.8",
             "figures/grid_T0.85_conditional_N15_QED0.8.png")
rows = [["N + QED, achieved mean QED", "T = 0.20", "T = 0.85"],
        ["request 0.4  /  0.6  /  0.8 at N = 15",
         "0.594 / 0.554 / 0.644", "0.514 / 0.538 / 0.593"],
        ["request 0.4  /  0.6  /  0.8 at N = 23",
         "0.582 / 0.573 / 0.590", "0.469 / 0.488 / 0.523"],
        ["request 0.4  /  0.6  /  0.8 at N = 30",
         "0.453 / 0.469 / 0.523", "0.342 / 0.367 / 0.400"],
        ["monotonic in the request?", "1 of 3 sizes", "3 of 3 sizes"]]
table(s, L, Inches(4.70), Inches(11.9), rows, widths=[4.9, 3.5, 3.5],
      row_h=Inches(0.33), body_size=11)
takeaway(s, "T = 0.20 gives higher absolute QED; T = 0.85 gives an ordered response. "
            "Pick the temperature for the question.", y=Inches(6.56))
page(s)

# =============================================================== slide 8 =====
s = prs.slides.add_slide(BLANK)
chrome(s, "06 · ZINC RESULTS", "Conditioning also moves chemistry toward ZINC",
       "Mean aromatic rings per valid molecule, against a size-matched training "
       "reference")

data = CategoryChartData()
data.categories = ["N = 15", "N = 23", "N = 30"]
data.add_series("Unconditional", (0.495, 0.578, 0.619))
data.add_series("N + QED", (0.731, 1.127, 1.215))
data.add_series("Real ZINC (size-matched)", (0.846, 1.817, 2.767))
graphic = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, L, Inches(1.92),
                             Inches(6.9), Inches(4.3), data)
chart = graphic.chart
style_chart(chart)
chart.plots[0].gap_width = 70
for index, colour in enumerate((PURPLE, ROSE, RGBColor(0xC9, 0xC2, 0xD4))):
    chart.plots[0].series[index].format.fill.solid()
    chart.plots[0].series[index].format.fill.fore_color.rgb = colour
chart.value_axis.tick_labels.font.size = Pt(9)
chart.category_axis.tick_labels.font.size = Pt(10)

panel(s, Inches(7.72), Inches(1.92), Inches(4.99), Inches(4.3))
frame = tb(s, Inches(7.98), Inches(2.08), Inches(4.5), Inches(0.3))
put(frame, "GAP TO THE REFERENCE, CLOSED", 10.5, LABEL, bold=True, first=True)
rows = [["Size", "Closed by conditioning"],
        ["N = 15", "67%"],
        ["N = 23", "44%"],
        ["N = 30", "28%"]]
table(s, Inches(7.98), Inches(2.44), Inches(4.5), rows, widths=[2.0, 2.5],
      row_h=Inches(0.33), body_size=11.5)
frame = tb(s, Inches(7.98), Inches(3.96), Inches(4.5), Inches(2.2))
put(frame, "Aromatic rings roughly double at N = 23 and N = 30 without being asked "
           "for.", 11.5, PLUM, bold=True, first=True, line=1.12, space_after=10)
put(frame, "QED has aromatic-ring count among its eight components, so this is the "
           "property pushing structure, not a free lunch.", 11, BODY, line=1.12,
    space_after=10)
put(frame, "Mean QED itself also moves toward the reference: 0.427 -> 0.482 at "
           "N = 30 (reference 0.591).", 11, BODY, line=1.12)
takeaway(s, "Asking for drug-likeness produces more drug-like chemistry, not only a "
            "better score.")
page(s)

# =============================================================== slide 9 =====
s = prs.slides.add_slide(BLANK)
chrome(s, "06 · ZINC RESULTS", "Where this stands",
       "ZINC250k generation quality, 10,000 attempts x 5 sampling repetitions, "
       "T = 0.85")
rows = [["Metric (%)", "MoFlow (published)", "Unconditional", "N + QED"],
        ["Validity after correction", "100.00", "98.85 ± 0.07", "99.12 ± 0.06"],
        ["Uniqueness among valid", "99.99", "100.00 ± 0.00", "100.00 ± 0.00"],
        ["Novelty among unique valid", "100.00", "100.00 ± 0.00", "100.00 ± 0.00"],
        ["N.U.V. over attempts", "99.99", "98.85 ± 0.07", "99.12 ± 0.05"]]
table(s, L, Inches(1.86), Inches(11.9), rows, widths=[4.0, 2.6, 2.65, 2.65],
      row_h=Inches(0.37))
frame = tb(s, L, Inches(3.84), Inches(11.9), Inches(0.26))
put(frame, "MoFlow-STYLE SAMPLING, NOT A REPRODUCTION", 10.5, LABEL, bold=True,
    first=True)
frame = tb(s, L, Inches(4.10), Inches(11.9), Inches(0.5))
put(frame, "Same 10,000 x 5 budget and the same T = 0.85 convention. Different "
           "representation (embedding-space flow plus a learned decoder), different "
           "atom vocabulary, and a different correction procedure.", 11.5, BODY,
    first=True, line=1.1)

for index, (head, text) in enumerate([
    ("Property targeting",
     "QED error down 12.4% on the fixed grid and 16.6% on percentile targets; hit "
     "yield up to 2x at the hardest cell."),
    ("Generation quality",
     "99.12% validity after assembly, with uniqueness and novelty at ceiling."),
    ("Chemistry",
     "Aromatic-ring counts move 28-67% of the way to a size-matched ZINC reference.")]):
    y = Inches(4.78) + index * Inches(0.58)
    frame = tb(s, L, y, Inches(2.9), Inches(0.3))
    put(frame, head, 13, ROSE, bold=True, first=True)
    frame = tb(s, Inches(3.70), y, Inches(8.9), Inches(0.5))
    put(frame, text, 12, BODY, first=True, line=1.1)
takeaway(s, "Conditioning on N and QED improves targeting and chemistry at no cost "
            "to generation quality.", y=Inches(6.62))
page(s)

# ============================================================== appendix =====
s = prs.slides.add_slide(BLANK)
chrome(s, "ZINC · APPENDIX", "Raw decoder output and the assembly step",
       "T = 0.85; the same 50,000 attempts per model")
rows = [["Metric (%)", "Unconditional", "N + QED"],
        ["Raw argmax validity", "3.90 ± 0.12", "5.58 ± 0.19"],
        ["After degree-constrained assembly", "98.85 ± 0.07", "99.12 ± 0.06"]]
table(s, L, Inches(1.90), Inches(11.9), rows, widths=[5.0, 3.45, 3.45],
      row_h=Inches(0.40))
frame = tb(s, L, Inches(3.30), Inches(11.9), Inches(1.0))
put(frame, "The headline validity depends on the assembly pipeline: an exact "
           "degree-constrained integer program over the decoder's bond logits, which "
           "requires a single connected fragment and deletes no atoms.", 12.5, BODY,
    first=True, line=1.15)
takeaway(s, "Conditioning improves raw output too (3.90% -> 5.58%), but the headline "
            "number is an assembled number.", y=Inches(6.62))
page(s)

s = prs.slides.add_slide(BLANK)
chrome(s, "ZINC · APPENDIX", "Where the distribution still differs",
       "Grid outputs at T = 0.20, mean descriptors at N = 30")
rows = [["N = 30 property", "Real ZINC", "Unconditional", "N + QED"],
        ["Perceived rings (SSSR)", "3.98", "8.12", "9.67"],
        ["Aromatic rings", "2.77", "0.62", "1.22"],
        ["Molecular weight", "424.5", "408.1", "408.6"],
        ["logP", "3.32", "2.01", "2.07"],
        ["QED", "0.591", "0.427", "0.482"]]
table(s, L, Inches(1.90), Inches(11.9), rows, widths=[4.4, 2.5, 2.5, 2.5],
      row_h=Inches(0.37))
frame = tb(s, L, Inches(4.26), Inches(11.9), Inches(1.4))
put(frame, "Excess perceived rings is the clearest remaining gap, and conditioning "
           "makes it larger, not smaller. SSSR counts every smallest ring, so fused "
           "and caged graphs inflate it quickly.", 12.5, BODY, first=True, line=1.15,
    space_after=10)
put(frame, "Aromatic rings, molecular weight, logP and QED all move toward the "
           "reference.", 12.5, PLUM, bold=True, line=1.15)
takeaway(s, "Validity does not imply a ZINC-like distribution; ring topology remains "
            "the open problem.", y=Inches(6.62))
page(s)

s = prs.slides.add_slide(BLANK)
chrome(s, "ZINC · APPENDIX", "Every fixed-grid cell",
       "T = 0.20; 2,000 attempts per target per repetition, five repetitions")
rows = [["N", "QED", "Uncond. mean", "N+QED mean", "Uncond. hit", "N+QED hit"],
        ["15", "0.4", "0.593", "0.594", "19.5%", "20.5%"],
        ["15", "0.6", "0.592", "0.554", "52.4%", "51.7%"],
        ["15", "0.8", "0.590", "0.644", "24.1%", "36.3%"],
        ["23", "0.4", "0.558", "0.582", "26.1%", "27.4%"],
        ["23", "0.6", "0.560", "0.573", "31.5%", "35.3%"],
        ["23", "0.8", "0.556", "0.590", "27.4%", "28.6%"],
        ["30", "0.4", "0.426", "0.453", "33.8%", "38.2%"],
        ["30", "0.6", "0.432", "0.469", "29.7%", "32.6%"],
        ["30", "0.8", "0.423", "0.523", "7.5%", "15.3%"]]
table(s, L, Inches(1.86), Inches(11.9), rows,
      widths=[1.1, 1.3, 2.5, 2.5, 2.25, 2.25], row_h=Inches(0.345),
      body_size=11)
takeaway(s, "The unconditional mean is flat down each size block; the conditioned "
            "mean tracks the request.", y=Inches(5.62))
page(s)

s = prs.slides.add_slide(BLANK)
chrome(s, "ZINC · APPENDIX", "Protocol and limits",
       "What this comparison does and does not establish")
rows = [["Question", "Answer from these runs"],
        ["Is the comparison matched?",
         "Same backbone, decoder, data split, 50,000 steps, shared size requests and "
         "sampling seeds."],
        ["Is it conditioning alone?",
         "No. The conditioned arm also carries the QED contrastive objective; the two "
         "are not separated here."],
        ["Training-seed robustness?",
         "Not established: one trained checkpoint per arm. The +/- values are sampling "
         "repetitions."],
        ["Best temperature?",
         "T = 0.20 for absolute QED, T = 0.85 for an ordered response; no matched sweep "
         "beyond these two."],
        ["Interval-valued requests?",
         "Not used. Requests are scalar; +/-0.1 and +/-0.01 are evaluator tolerances "
         "applied after generation."]]
table(s, L, Inches(1.86), Inches(11.9), rows, widths=[3.5, 8.4],
      row_h=Inches(0.62), body_size=11.5)
takeaway(s, "Next question worth answering: does the contrastive term alone, without "
            "QED in the prior, give the same response?", y=Inches(6.00))
page(s)

prs.save(OUT)
print("saved %s  (%d slides)" % (OUT, len(prs.slides.__iter__.__self__._sldIdLst)))
