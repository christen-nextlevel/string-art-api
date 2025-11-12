# pipeline/csv_to_pdf_nopandas.py
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
import csv
from pathlib import Path

def csv_to_pdf(csv_file: str, pdf_file: str,
               title="String Art Threading Instructions",
               subtitle="Nail 1 is at the top (12 o'clock), numbers increase clockwise.",
               page_size=A4, landscape_mode=False,
               margins_mm=(15,15,18,18),
               header_font=11, row_font=9,
               rows_per_page=0,
               col_widths_mm=(25,45,45)):
    rows = []
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        field_map = {k.lower().strip(): k for k in reader.fieldnames or []}
        fn_key = field_map.get("from_nail") or field_map.get("from") or list(field_map.values())[0]
        tn_key = field_map.get("to_nail") or field_map.get("to") or list(field_map.values())[1]
        for i, row in enumerate(reader, start=1):
            rows.append([i, str(row[fn_key]).strip(), str(row[tn_key]).strip()])

    if not rows:
        raise ValueError(f"No rows found in {csv_file}")

    pagesize = landscape(page_size) if landscape_mode else page_size
    left, right, top, bottom = (x * mm for x in margins_mm)
    Path(pdf_file).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(pdf_file, pagesize=pagesize,
                            leftMargin=left, rightMargin=right,
                            topMargin=top, bottomMargin=bottom)

    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(subtitle, styles["Normal"]))
    story.append(Spacer(1, 10))

    header = ["Step", "From Nail", "To Nail"]
    col_widths = [w * mm for w in col_widths_mm]

    chunks = [rows] if not rows_per_page or rows_per_page <= 0 else \
             [rows[i:i+rows_per_page] for i in range(0, len(rows), rows_per_page)]

    for idx, chunk in enumerate(chunks):
        table = Table([header] + chunk, colWidths=col_widths)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.black),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTSIZE", (0, 0), (-1, 0), header_font),
            ("FONTSIZE", (0, 1), (-1, -1), row_font),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ]))
        story.append(table)
        if idx < len(chunks) - 1:
            story.append(PageBreak())

    doc.build(story)
    return pdf_file