"""Build docs/report.pdf from docs/report.md and embed the generated screenshots."""
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "report.md"
OUT = ROOT / "docs" / "report.pdf"
SHOTS = ROOT / "docs" / "screenshots"

styles = getSampleStyleSheet()
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.5, leading=13,
                      spaceAfter=6)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=7.5, leading=9.5, spaceAfter=0)
CODE = ParagraphStyle("Code", parent=styles["Code"], fontSize=7.5, leading=9.5)


def inline(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("—", "-").replace("→", "->").replace("×", "x")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    return text


def table_from(lines, i):
    rows = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            rows.append([Paragraph(inline(c), CELL) for c in cells])
        i += 1
    table = Table(rows, hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dddddd")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table, i


def build_story():
    lines = SRC.read_text().splitlines()
    story = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("```"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            story.append(Preformatted("\n".join(code), CODE))
            story.append(Spacer(1, 4))
            continue
        if stripped.startswith("|"):
            table, i = table_from(lines, i)
            story.append(table)
            story.append(Spacer(1, 8))
            continue
        if stripped.startswith("### "):
            story.append(Paragraph(inline(stripped[4:]), styles["Heading3"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(inline(stripped[3:]), styles["Heading2"]))
        elif stripped.startswith("# "):
            story.append(Paragraph(inline(stripped[2:]), styles["Title"]))
        elif stripped.startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(Paragraph("&bull; " + inline(lines[i].strip()[2:]), BODY))
                i += 1
            story.extend(items)
            continue
        else:
            para = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(
                    r"^(#|\||-|```|\d+\. )", lines[i].strip()):
                para.append(lines[i].strip())
                i += 1
            story.append(Paragraph(inline(" ".join(para)), BODY))
            continue
        i += 1

    story.append(Paragraph("Figures", styles["Heading1"]))
    captions = {
        "q1_teacher_curve.png": "Teacher training and validation accuracy (CIFAR-100).",
        "q1_baseline_vs_kd.png": "Distilled vs baseline validation accuracy for each student.",
        "q2_nas_search.png": "NAS candidates: accuracy vs measured CPU latency.",
        "onnx_benchmark.png": "ONNX Runtime CPU inference latency for all exported models.",
    }
    for name, caption in captions.items():
        path = SHOTS / name
        if not path.exists():
            continue
        image = Image(str(path))
        max_w = 170 * mm
        if image.drawWidth > max_w:
            scale = max_w / image.drawWidth
            image.drawWidth *= scale
            image.drawHeight *= scale
        story.append(image)
        story.append(Paragraph(inline(caption), BODY))
        story.append(Spacer(1, 8))
    return story


if __name__ == "__main__":
    SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                      topMargin=18 * mm, bottomMargin=18 * mm,
                      title="Assignment 4 Technical Report").build(build_story())
    print("Wrote", OUT)
