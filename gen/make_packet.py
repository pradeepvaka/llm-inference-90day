#!/usr/bin/env python3
"""Build a daily packet PDF from a markdown-ish content file.

Usage:
    python3 gen/make_packet.py --input day01.md --output day01.pdf \
        --title "The LLM inference stack" --day 1 --date 2026-09-09

Supported markdown-ish syntax: # / ## / ### headings, paragraphs,
`-`/`*` bullets, fenced ``` code blocks, simple | tables |.
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


# ---------------------------------------------------------------- parsing

def parse_md(text):
    """Parse markdown-ish text into a list of blocks.

    Block kinds: ('h1', text), ('h2', text), ('h3', text), ('p', text),
    ('bullet', text), ('code', text), ('table', rows).
    """
    blocks = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # fenced code block
        if stripped.startswith("```"):
            buf = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            blocks.append(("code", "\n".join(buf)))
            continue

        # simple pipe table
        if stripped.startswith("|") and stripped.endswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                # skip separator rows like |---|---|
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in row):
                    rows.append(row)
                i += 1
            if rows:
                blocks.append(("table", rows))
            continue

        # headings
        if stripped.startswith("# "):
            blocks.append(("h1", stripped[2:].strip()))
        elif stripped.startswith("## "):
            blocks.append(("h2", stripped[3:].strip()))
        elif stripped.startswith("### "):
            blocks.append(("h3", stripped[4:].strip()))
        elif stripped.startswith(("- ", "* ")):
            blocks.append(("bullet", stripped[2:].strip()))
        elif stripped == "":
            pass
        else:
            # paragraph: join wrapped lines
            buf = [stripped]
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if nxt == "" or nxt.startswith(("#", "- ", "* ", "```", "|")):
                    break
                buf.append(nxt)
                i += 1
            blocks.append(("p", " ".join(buf)))
            continue
        i += 1
    return blocks


# ---------------------------------------------------------------- building

def inline_fmt(text):
    """Minimal inline formatting: **bold**, `code`."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    return text


def make_styles(scale):
    base = 11 * scale
    return {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold",
                                fontSize=22 * scale, leading=26 * scale,
                                spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica",
                                   fontSize=13 * scale, leading=16 * scale,
                                   textColor=colors.HexColor("#444444"),
                                   spaceAfter=2),
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold",
                             fontSize=16 * scale, leading=19 * scale,
                             spaceBefore=14, spaceAfter=6),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold",
                             fontSize=13.5 * scale, leading=16 * scale,
                             spaceBefore=10, spaceAfter=4),
        "h3": ParagraphStyle("h3", fontName="Helvetica-BoldOblique",
                             fontSize=base, leading=base + 3,
                             spaceBefore=8, spaceAfter=3),
        "p": ParagraphStyle("p", fontName="Helvetica", fontSize=base,
                            leading=base + 4, spaceAfter=5,
                            alignment=4),  # justified
        "bullet": ParagraphStyle("bullet", fontName="Helvetica",
                                 fontSize=base, leading=base + 4,
                                 leftIndent=18, spaceAfter=3,
                                 bulletIndent=8),
        "code": ParagraphStyle("code", fontName="Courier",
                               fontSize=9 * scale, leading=12 * scale),
        "cell": ParagraphStyle("cell", fontName="Helvetica",
                               fontSize=9.5 * scale, leading=12 * scale),
        "cellh": ParagraphStyle("cellh", fontName="Helvetica-Bold",
                                fontSize=9.5 * scale, leading=12 * scale),
    }


CODE_BG = colors.HexColor("#F4F4F4")


def build_flowables(blocks, styles):
    fl = []

    def para(txt, style):
        return Paragraph(inline_fmt(txt), style)

    for kind, payload in blocks:
        if kind == "h1":
            fl.append(KeepTogether([
                Spacer(1, 6),
                para(payload, styles["h1"]),
                HRFlowable(width="100%", thickness=1,
                           color=colors.HexColor("#CCCCCC")),
                Spacer(1, 2),
            ]))
        elif kind == "h2":
            fl.append(para(payload, styles["h2"]))
        elif kind == "h3":
            fl.append(para(payload, styles["h3"]))
        elif kind == "bullet":
            fl.append(Paragraph(inline_fmt(payload), styles["bullet"],
                               bulletText="\u2022"))
        elif kind == "code":
            pre = Preformatted(payload, styles["code"])
            t = Table([[pre]], colWidths=[7.0 * inch])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            fl.append(Spacer(1, 3))
            fl.append(t)
            fl.append(Spacer(1, 5))
        elif kind == "table":
            rows = payload
            ncols = max(len(r) for r in rows)
            rows = [r + [""] * (ncols - len(r)) for r in rows]
            styled = [[Paragraph(inline_fmt(c), styles["cellh"]) for c in rows[0]]]
            styled += [[Paragraph(inline_fmt(c), styles["cell"]) for c in r]
                       for r in rows[1:]]
            cw = (7.0 * inch) / ncols
            t = Table(styled, colWidths=[cw] * ncols, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#AAAAAA")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            fl.append(Spacer(1, 3))
            fl.append(t)
            fl.append(Spacer(1, 5))
        else:  # paragraph
            fl.append(para(payload, styles["p"]))
    return fl


class PageCounter:
    def __init__(self):
        self.pages = 0

    def __call__(self, canvas, doc):
        self.pages = canvas.getPageNumber()
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(0.75 * inch, 0.6 * inch, REPO_FOOTER)
        canvas.drawRightString(7.75 * inch, 0.6 * inch,
                               f"Page {canvas.getPageNumber()}")
        canvas.restoreState()


def build_pdf(blocks, day, title, date, scale, dest=None):
    """Build the packet. dest=None renders to a temp buffer (for page count),
    otherwise to the given file path. Returns (page_count, style_scale)."""
    styles = make_styles(scale)
    fl = []

    # Title header block (only on page 1; kept short)
    fl.append(Paragraph(inline_fmt(f"Day {day}: {title}"), styles["title"]))
    fl.append(Paragraph(f"LLM Inference 90-Day &nbsp;&bull;&nbsp; {date}",
                        styles["subtitle"]))
    fl.append(HRFlowable(width="100%", thickness=2,
                         color=colors.HexColor("#222222"), spaceAfter=8))

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
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        blocks = parse_md(f.read())

    pages = None
    chosen_scale = 1.0
    for scale in (1.0, 0.92, 0.85):
        pages = build_pdf(blocks, args.day, args.title, args.date, scale)
        chosen_scale = scale
        if pages <= MAX_PAGES:
            break

    trimmed = 0
    # Graceful trim: drop trailing plain paragraphs first (keep >= 1 page
    # worth of content). Prefer dropping from the tail.
    while pages > MAX_PAGES and len(blocks) > 3:
        # find last droppable paragraph/bullet index
        for idx in range(len(blocks) - 1, -1, -1):
            if blocks[idx][0] in ("p", "bullet"):
                del blocks[idx]
                trimmed += 1
                break
        else:
            break
        pages = build_pdf(blocks, args.day, args.title, args.date,
                          chosen_scale)

    if pages > MAX_PAGES:
        print(f"ERROR: still {pages} pages after trimming; "
              f"shorten the input.", file=sys.stderr)
        sys.exit(2)

    final_pages = build_pdf(blocks, args.day, args.title, args.date,
                            chosen_scale, dest=args.output)

    print(f"Wrote {args.output}: {final_pages} page(s), "
          f"font scale {chosen_scale:.2f}")
    if chosen_scale < 1.0:
        print(f"WARNING: shrunk fonts to {chosen_scale:.0%} to fit "
              f"{MAX_PAGES} pages.")
    if trimmed:
        print(f"WARNING: trimmed {trimmed} trailing paragraph(s) to fit "
              f"{MAX_PAGES} pages.")


if __name__ == "__main__":
    main()
