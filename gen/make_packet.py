#!/usr/bin/env python3
"""Build a daily packet PDF from a markdown-ish content file.

Usage:
    python3 gen/make_packet.py --input day01.md --output day01.pdf \
        --title "The LLM inference stack" --day 1 --date 2026-09-09

Supported markdown-ish syntax:
    # / ## / ###   headings (content should NOT repeat the title as #)
    > ...          callout box (Time / Prereqs / key warnings)
    - / *          bullet list
    1. / 2. ...   ordered list (indented continuation lines join the item)
    ```            fenced code blocks (auto-wrapped, never mid-word)
    | a | b |      pipe tables (proportional columns, zebra rows)
    Table: ...     caption for the preceding table
    **bold**, *italic*, `code` inline

Enforces a hard max of 10 pages by shrinking fonts, then trimming trailing
body paragraphs (with a stdout warning).
"""

import argparse
import io
import re
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

MAX_PAGES = 10
REPO_FOOTER = "LLM Inference 90-Day"
PAGE_W = 7.0 * inch  # LETTER width minus 0.75in margins on both sides

ACCENT = colors.HexColor("#1D4ED8")      # deep blue
INK = colors.HexColor("#111827")        # near-black
MUTED = colors.HexColor("#6B7280")       # gray
HEADER_BG = colors.HexColor("#1F2937")   # dark slate (table headers)
ZEBRA_BG = colors.HexColor("#F5F7FA")    # alternating row tint
GRID = colors.HexColor("#D1D5DB")
CODE_BG = colors.HexColor("#F3F4F6")
CALLOUT_BG = colors.HexColor("#EFF6FF")


# ---------------------------------------------------------------- parsing

def parse_md(text):
    """Parse markdown-ish text into blocks.

    Kinds: ('h1',t) ('h2',t) ('h3',t) ('p',t) ('bullet',t)
           ('olist', [(num, text), ...])  -- text may contain \\n continuations
           ('code', text) ('table', rows) ('caption', t) ('quote', t)
    """
    blocks = []
    lines = text.splitlines()
    i = 0
    n = len(lines)

    def is_list_start(s):
        return s.startswith(("- ", "* ")) or re.match(r"\d+\.\s", s)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # fenced code block
        if stripped.startswith("```"):
            buf = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i].rstrip())
                i += 1
            i += 1  # skip closing fence
            blocks.append(("code", "\n".join(buf).strip("\n")))
            continue

        # pipe table
        if stripped.startswith("|") and stripped.endswith("|"):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in row):
                    rows.append(row)
                i += 1
            if rows:
                blocks.append(("table", rows))
            continue

        # table caption
        if stripped.startswith("Table:"):
            blocks.append(("caption", stripped[len("Table:"):].strip()))
            i += 1
            continue

        # callout
        if stripped.startswith("> "):
            buf = [stripped[2:].strip()]
            i += 1
            while i < n and lines[i].strip().startswith("> "):
                buf.append(lines[i].strip()[2:].strip())
                i += 1
            blocks.append(("quote", " ".join(buf)))
            continue

        # headings
        if stripped.startswith("# "):
            blocks.append(("h1", stripped[2:].strip()))
            i += 1
            continue
        if stripped.startswith("## "):
            blocks.append(("h2", stripped[3:].strip()))
            i += 1
            continue
        if stripped.startswith("### "):
            blocks.append(("h3", stripped[4:].strip()))
            i += 1
            continue

        # lists (bullets and ordered), with indented continuations
        if is_list_start(stripped):
            items = []  # (kind, num_or_None, text)
            while i < n:
                s = lines[i].strip()
                m = re.match(r"(\d+)\.\s+(.*)", s)
                if m:
                    items.append(("olist", m.group(1), m.group(2)))
                elif s.startswith(("- ", "* ")):
                    items.append(("bullet", None, s[2:].strip()))
                elif s == "" or s.startswith(("#", "```", "|", "> ")):
                    break
                elif re.match(r"\s{2,}\S", lines[i]) and items:
                    # indented continuation of the previous item
                    k, num, txt = items[-1]
                    items[-1] = (k, num, txt + "\n" + s)
                else:
                    break
                i += 1
            # group consecutive same-kind runs
            run = []
            for k, num, txt in items:
                if run and run[0][0] != k:
                    blocks.append(_flush_list(run))
                    run = []
                run.append((k, num, txt))
            if run:
                blocks.append(_flush_list(run))
            continue

        if stripped == "":
            i += 1
            continue

        # paragraph: join wrapped lines
        buf = [stripped]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if (nxt == "" or nxt.startswith(("#", "- ", "* ", "```", "|", "> "))
                    or nxt.startswith("Table:") or re.match(r"\d+\.\s", nxt)):
                break
            buf.append(nxt)
            i += 1
        blocks.append(("p", " ".join(buf)))

    return blocks


def _flush_list(run):
    kind = run[0][0]
    if kind == "olist":
        return ("olist", [(num, txt) for _, num, txt in run])
    return ("bullets", [txt for _, _, txt in run])


# ---------------------------------------------------------------- building

def inline_fmt(text):
    """**bold**, *italic* (space-aware, skips math like 2 * 80), `code`."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    # *italic*: opening * must follow start/whitespace/open-punct and a
    # non-space; closing * must precede whitespace/end/punctuation. This
    # keeps math like 2 * 80 untouched.
    text = re.sub(
        r"(?:(?<=\s)|^|(?<=[(\[\"']))\*(\S(?:[^*]*\S)?)\*(?=\s|$|[.,;:!?)\]\"'])",
        r"<i>\1</i>", text)
    # line continuations inside list items
    text = text.replace("\n", "<br/>")
    return text


def make_styles(scale):
    base = 11 * scale
    return {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold",
                                fontSize=24 * scale, leading=28 * scale,
                                textColor=INK, spaceAfter=2),
        "eyebrow": ParagraphStyle("eyebrow", fontName="Helvetica-Bold",
                                  fontSize=9 * scale, leading=11 * scale,
                                  textColor=ACCENT, spaceAfter=6),
        "deck": ParagraphStyle("deck", fontName="Helvetica",
                               fontSize=12 * scale, leading=15 * scale,
                               textColor=MUTED, spaceAfter=0),
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold",
                             fontSize=17 * scale, leading=20 * scale,
                             textColor=INK, spaceBefore=16, spaceAfter=6),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold",
                             fontSize=14 * scale, leading=17 * scale,
                             textColor=HEADER_BG, spaceBefore=14,
                             spaceAfter=5),
        "h3": ParagraphStyle("h3", fontName="Helvetica-BoldOblique",
                             fontSize=base, leading=base + 3,
                             textColor=INK, spaceBefore=9, spaceAfter=3),
        "p": ParagraphStyle("p", fontName="Helvetica", fontSize=base,
                            leading=base + 4.5, spaceAfter=6, textColor=INK),
        "bullet": ParagraphStyle("bullet", fontName="Helvetica",
                                 fontSize=base, leading=base + 4.5,
                                 leftIndent=20, spaceAfter=3,
                                 bulletIndent=9, textColor=INK),
        "olist": ParagraphStyle("olist", fontName="Helvetica",
                                fontSize=base, leading=base + 4.5,
                                leftIndent=24, spaceAfter=4,
                                bulletIndent=9, textColor=INK),
        "code": ParagraphStyle("code", fontName="Courier",
                               fontSize=9.5 * scale,
                               leading=13 * scale, textColor=INK),
        "cell": ParagraphStyle("cell", fontName="Helvetica",
                               fontSize=9.5 * scale, leading=12.5 * scale,
                               textColor=INK),
        "cellh": ParagraphStyle("cellh", fontName="Helvetica-Bold",
                                fontSize=9.5 * scale, leading=12.5 * scale,
                                textColor=colors.white),
        "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique",
                                   fontSize=9 * scale, leading=11 * scale,
                                   textColor=MUTED, spaceBefore=2,
                                   spaceAfter=8),
        "quote": ParagraphStyle("quote", fontName="Helvetica",
                                fontSize=10.5 * scale,
                                leading=14 * scale, textColor=INK),
    }


def wrap_code(text, font_name, font_size, max_width):
    """Wrap code lines at spaces; split unbreakable tokens mid-word."""
    out = []
    for line in text.split("\n"):
        if stringWidth(line, font_name, font_size) <= max_width:
            out.append(line)
            continue
        words, cur = line.split(" "), ""
        for w in words:
            trial = (cur + " " + w).strip()
            if stringWidth(trial, font_name, font_size) <= max_width:
                cur = trial
            else:
                if cur:
                    out.append(cur)
                # word itself too long: hard-split it
                while stringWidth(w, font_name, font_size) > max_width:
                    k = len(w)
                    while k > 1 and stringWidth(w[:k], font_name,
                                               font_size) > max_width:
                        k -= 1
                    out.append(w[:k])
                    w = w[k:]
                cur = w
        if cur:
            out.append(cur)
    return "\n".join(out)


def proportional_widths(rows, cell_style, header_style, total):
    """Column widths proportional to content, clamped to sane bounds."""
    ncols = max(len(r) for r in rows)
    maxw = [0.0] * ncols
    for ri, row in enumerate(rows):
        for ci in range(ncols):
            txt = row[ci] if ci < len(row) else ""
            # strip markdown for measuring
            plain = re.sub(r"\*\*(.+?)\*\*", r"\1", txt)
            plain = re.sub(r"`(.+?)`", r"\1", plain)
            style = header_style if ri == 0 else cell_style
            w = stringWidth(plain, style.fontName, style.fontSize)
            maxw[ci] = max(maxw[ci], w)
    # clamp each column, then scale to fit
    lo, hi = 0.9 * inch, 3.6 * inch
    clamped = [min(max(w + 14, lo), hi) for w in maxw]
    s = sum(clamped)
    if s > total:
        clamped = [w * total / s for w in clamped]
    return clamped


def clean_title(title, day):
    t = title.strip()
    for pat in (f"Day {day} \u2014", f"Day {day} -", f"Day {day}:"):
        if t.lower().startswith(pat.lower()):
            return t[len(pat):].strip()
    return t


def build_flowables(blocks, styles):
    fl = []

    def para(txt, style):
        return Paragraph(inline_fmt(txt), style)

    i = 0
    while i < len(blocks):
        kind, payload = blocks[i]

        if kind == "h1":
            fl.append(KeepTogether([
                para(payload, styles["h1"]),
                HRFlowable(width="100%", thickness=1, color=GRID),
                Spacer(1, 2),
            ]))
        elif kind == "h2":
            fl.append(KeepTogether([
                para(payload, styles["h2"]),
                HRFlowable(width="100%", thickness=0.75,
                           color=colors.HexColor("#E5E7EB")),
            ]))
        elif kind == "h3":
            fl.append(para(payload, styles["h3"]))
        elif kind == "bullets":
            for txt in payload:
                fl.append(Paragraph(inline_fmt(txt), styles["bullet"],
                                    bulletText="\u2022"))
        elif kind == "olist":
            for num, txt in payload:
                fl.append(Paragraph(inline_fmt(txt), styles["olist"],
                                    bulletText=f"{num}."))
        elif kind == "quote":
            cell = Paragraph(inline_fmt(payload), styles["quote"])
            t = Table([[cell]], colWidths=[PAGE_W - 14])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), CALLOUT_BG),
                ("LINEBEFORE", (0, 0), (0, -1), 3.5, ACCENT),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]))
            fl.append(Spacer(1, 4))
            fl.append(t)
            fl.append(Spacer(1, 8))
        elif kind == "code":
            fs = styles["code"].fontSize
            wrapped = wrap_code(payload, "Courier", fs, PAGE_W - 20)
            pre = Preformatted(wrapped, styles["code"])
            t = Table([[pre]], colWidths=[PAGE_W])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
                ("ROUNDEDCORNERS", [3, 3, 3, 3]),
                ("BOX", (0, 0), (-1, -1), 0.5, GRID),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]))
            fl.append(Spacer(1, 4))
            fl.append(t)
            fl.append(Spacer(1, 8))
        elif kind == "table":
            rows = payload
            ncols = max(len(r) for r in rows)
            rows = [r + [""] * (ncols - len(r)) for r in rows]
            styled = [[Paragraph(inline_fmt(c), styles["cellh"])
                       for c in rows[0]]]
            styled += [[Paragraph(inline_fmt(c), styles["cell"]) for c in r]
                       for r in rows[1:]]
            widths = proportional_widths(rows, styles["cell"],
                                         styles["cellh"], PAGE_W)
            t = Table(styled, colWidths=widths, repeatRows=1)
            style_cmds = [
                ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
            for ri in range(1, len(styled)):
                if ri % 2 == 0:
                    style_cmds.append(
                        ("BACKGROUND", (0, ri), (-1, ri), ZEBRA_BG))
            t.setStyle(TableStyle(style_cmds))
            fl.append(Spacer(1, 4))
            fl.append(t)
            # attach a following caption to the table
            if i + 1 < len(blocks) and blocks[i + 1][0] == "caption":
                fl.append(para(blocks[i + 1][1], styles["caption"]))
                i += 1
            else:
                fl.append(Spacer(1, 8))
        elif kind == "caption":
            fl.append(para(payload, styles["caption"]))
        else:  # paragraph
            fl.append(para(payload, styles["p"]))
        i += 1
    return fl


class PageCounter:
    def __init__(self):
        self.pages = 0

    def __call__(self, canvas, doc):
        self.pages = canvas.getPageNumber()
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.75 * inch, 0.6 * inch, REPO_FOOTER)
        canvas.drawRightString(7.75 * inch, 0.6 * inch,
                               f"Page {canvas.getPageNumber()}")
        canvas.restoreState()


def build_pdf(blocks, day, title, date, scale, dest=None,
              minutes="~75 minutes"):
    """Build the packet. dest=None renders to a buffer (page count check).
    Returns (page_count, style_scale)."""
    styles = make_styles(scale)
    fl = []

    title = clean_title(title, day)
    fl.append(Paragraph(
        f"LLM INFERENCE 90-DAY &nbsp;&bull;&nbsp; {date}", styles["eyebrow"]))
    fl.append(Paragraph(inline_fmt(title), styles["title"]))
    fl.append(Paragraph(f"Day {day} of 90 &nbsp;&bull;&nbsp; about {minutes}",
                        styles["deck"]))
    fl.append(HRFlowable(width="100%", thickness=2, color=INK, spaceAfter=10,
                         spaceBefore=6))

    fl.extend(build_flowables(blocks, styles))

    counter = PageCounter()
    doc = BaseDocTemplate(dest if dest else io.BytesIO(),
                          pagesize=LETTER,
                          leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                          topMargin=0.75 * inch, bottomMargin=0.9 * inch)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  id="main")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame],
                                       onPage=counter)])
    doc.build(fl)
    return counter.pages


def main():
    ap = argparse.ArgumentParser(description="Build a daily packet PDF.")
    ap.add_argument("--input", required=True, help="markdown-ish input file")
    ap.add_argument("--output", required=True, help="output PDF path")
    ap.add_argument("--title", required=True, help="day title")
    ap.add_argument("--day", required=True, type=int, help="day number")
    ap.add_argument("--date", required=True, help="packet date YYYY-MM-DD")
    ap.add_argument("--minutes", default="~75 minutes",
                    help="estimated time, shown under the title")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        blocks = parse_md(f.read())

    pages = None
    chosen_scale = 1.0
    for scale in (1.0, 0.92, 0.85):
        pages = build_pdf(blocks, args.day, args.title, args.date, scale,
                          minutes=args.minutes)
        chosen_scale = scale
        if pages <= MAX_PAGES:
            break

    trimmed = 0
    cut_sections = 0
    while pages > MAX_PAGES and len(blocks) > 3:
        dropped = False
        for kinds, counter_name in ((("p", "bullets"), "body"),
                                    (("h3", "h2", "h1"), "section")):
            for idx in range(len(blocks) - 1, -1, -1):
                if blocks[idx][0] in kinds:
                    del blocks[idx]
                    if counter_name == "body":
                        trimmed += 1
                    else:
                        cut_sections += 1
                    dropped = True
                    break
            if dropped:
                break
        if not dropped:
            break
        pages = build_pdf(blocks, args.day, args.title, args.date,
                          chosen_scale, minutes=args.minutes)

    if pages > MAX_PAGES:
        print(f"ERROR: still {pages} pages after trimming; "
              f"shorten the input.", file=sys.stderr)
        sys.exit(2)

    final_pages = build_pdf(blocks, args.day, args.title, args.date,
                            chosen_scale, dest=args.output,
                            minutes=args.minutes)

    print(f"Wrote {args.output}: {final_pages} page(s), "
          f"font scale {chosen_scale:.2f}")
    if chosen_scale < 1.0:
        print(f"WARNING: shrunk fonts to {chosen_scale:.0%} to fit "
              f"{MAX_PAGES} pages.")
    if trimmed:
        print(f"WARNING: trimmed {trimmed} trailing paragraph(s) to fit "
              f"{MAX_PAGES} pages.")
    if cut_sections:
        print(f"WARNING: cut {cut_sections} trailing section(s) to fit "
              f"{MAX_PAGES} pages; consider splitting the day's content.")


if __name__ == "__main__":
    main()
