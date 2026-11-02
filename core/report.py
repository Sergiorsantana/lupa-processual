"""Camada 6: relatório PDF para impressão (resumo, gráficos, achados com localização e páginas destacadas)."""
from __future__ import annotations

import io
import time
from xml.sax.saxutils import escape

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .config import SEVERIDADES
from .detectors import show_invisibles
from .viz import SEV_COLOR, SEV_LABEL, VERDICT_COLOR, highlight_page

LAYERS = [
    ("1. Ingestão multimodal", "PDF, DOCX, TXT e imagens. Extrai texto com cor, tamanho, opacidade e camada, renderiza páginas e roda OCR."),
    ("2. Detecção determinística", "Texto branco, fonte minúscula, fora da página, coberto por forma, modo invisível, Unicode invisível, Base64, LSB, baixo contraste."),
    ("3. Banco vetorial e RAG", "Chunks de texto, OCR e metadados com localização, indexados em coleção vetorial. Sondas semânticas consultam a coleção."),
    ("4. Agentes e orquestrador", "Agentes de texto, visual, metadados e semântico, seguidos de verificação cruzada no banco vetorial."),
    ("5. Juiz", "Pondera severidade, agente, visibilidade e corroboração. Agrega o risco e emite o veredito."),
    ("6. Apresentação", "Dashboard de conformidade, localização nos documentos e imagens e este relatório."),
]


def rl(text: str) -> str:
    """Prepara texto para o reportlab: mostra invisíveis, escapa XML e troca o que a fonte padrão não cobre."""
    t = show_invisibles(text or "")
    t = escape(t)
    return t.encode("latin-1", "replace").decode("latin-1")


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("H1x", parent=ss["Title"], fontSize=20, leading=24, spaceAfter=6, textColor=colors.HexColor("#1F2A44")))
    ss.add(ParagraphStyle("H2x", parent=ss["Heading2"], fontSize=14, leading=18, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#1F2A44")))
    ss.add(ParagraphStyle("H3x", parent=ss["Heading3"], fontSize=11, leading=14, spaceBefore=6, spaceAfter=2, textColor=colors.HexColor("#333333")))
    ss.add(ParagraphStyle("Bx", parent=ss["BodyText"], fontSize=9.5, leading=13))
    ss.add(ParagraphStyle("Sx", parent=ss["BodyText"], fontSize=7.8, leading=10))
    ss.add(ParagraphStyle("Cx", parent=ss["BodyText"], fontSize=8, leading=10, alignment=TA_CENTER))
    return ss


def _verdict_cell(verdict: str, st):
    col = VERDICT_COLOR.get(verdict, "#555555")
    ink = "#0b0b0b" if verdict == "SUSPEITO" else "#ffffff"
    return Paragraph(f'<font color="{ink}"><b>{rl(verdict)}</b></font>', ParagraphStyle("vc", parent=st["Cx"], backColor=colors.HexColor(col),
                                                                                         textColor=colors.HexColor(ink), leading=14, fontSize=8.5))


def _where(f) -> str:
    parts = []
    if f.locator:
        parts.append(f.locator)
    if f.bbox:
        x0, y0, x1, y1 = f.bbox
        parts.append(f"área ({x0:.0f}, {y0:.0f}) a ({x1:.0f}, {y1:.0f})")
    return "; ".join(parts) or "estrutura do arquivo"


def _severity_chart(counts: dict) -> Drawing:
    d = Drawing(240, 120)
    bc = VerticalBarChart()
    bc.x, bc.y, bc.width, bc.height = 30, 20, 190, 80
    bc.data = [[counts.get(s, 0) for s in SEVERIDADES]]
    bc.categoryAxis.categoryNames = [SEV_LABEL[s] for s in SEVERIDADES]
    bc.categoryAxis.labels.fontSize = 7
    bc.valueAxis.labels.fontSize = 7
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = max(max(bc.data[0]), 1) + 1
    bc.valueAxis.valueStep = max(1, (bc.valueAxis.valueMax // 4))
    bc.barLabelFormat = "%d"
    bc.barLabels.fontSize = 7
    bc.barLabels.nudge = 6
    bc.bars.strokeColor = None
    for i, s in enumerate(SEVERIDADES):
        bc.bars[(0, i)].fillColor = colors.HexColor(SEV_COLOR[s])
    d.add(bc)
    d.add(String(120, 108, "Achados por severidade", fontSize=8, textAnchor="middle"))
    return d


def _verdict_chart(docs: list) -> Drawing:
    d = Drawing(240, 120)
    order = ["CONFORME", "SUSPEITO", "NÃO CONFORME"]
    vals = [sum(1 for x in docs if x["verdict"] == v) for v in order]
    if sum(vals) == 0:
        return d
    pie = Pie()
    pie.x, pie.y, pie.width, pie.height = 70, 14, 90, 90
    pie.data = vals
    pie.labels = [f"{o.title()} ({n})" if n else "" for o, n in zip(order, vals)]
    pie.sideLabels = True
    pie.simpleLabels = False
    pie.slices.fontSize = 7
    for i, o in enumerate(order):
        pie.slices[i].fillColor = colors.HexColor(VERDICT_COLOR[o])
        pie.slices[i].strokeColor = colors.white
    d.add(pie)
    d.add(String(120, 108, "Documentos por veredito", fontSize=8, textAnchor="middle"))
    return d


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(1.8 * cm, 1.1 * cm, "Detecção de comandos ocultos em processo judicial: relatório de conformidade")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Página {doc.page}")
    canvas.restoreState()


def build_report(docs: list[dict], findings_by_doc: dict, params: dict, max_pages_per_doc: int = 6) -> bytes:
    """docs: linhas de documents; findings_by_doc: {doc_id: [Finding]}; params: parâmetros em uso."""
    st = _styles()
    buf = io.BytesIO()
    pdf = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.8 * cm,
                            title="Relatório de conformidade", author="Detector de comandos ocultos")
    W = A4[0] - 3.6 * cm
    story = []
    story.append(Paragraph("Relatório de conformidade", st["H1x"]))
    story.append(Paragraph("Detecção de comandos ocultos em processo judicial com RAG multimodal, agentes e juiz", st["Bx"]))
    story.append(Paragraph(f"Gerado em {time.strftime('%d/%m/%Y %H:%M')} | {len(docs)} documento(s) analisado(s)", st["Sx"]))
    story.append(Spacer(1, 8))

    all_f = [f for d in docs for f in findings_by_doc.get(d["id"], []) if f.status == "ativo"]
    counts = {s: sum(1 for f in all_f if f.severity == s) for s in SEVERIDADES}
    kpi = Table([[
        Paragraph(f"<b>{len(docs)}</b><br/>documentos", st["Cx"]),
        Paragraph(f"<b>{sum(1 for d in docs if d['verdict'] == 'CONFORME')}</b><br/>conformes", st["Cx"]),
        Paragraph(f"<b>{sum(1 for d in docs if d['verdict'] == 'SUSPEITO')}</b><br/>suspeitos", st["Cx"]),
        Paragraph(f"<b>{sum(1 for d in docs if d['verdict'] == 'NÃO CONFORME')}</b><br/>não conformes", st["Cx"]),
        Paragraph(f"<b>{len(all_f)}</b><br/>achados", st["Cx"]),
    ]], colWidths=[W / 5] * 5, rowHeights=[1.3 * cm])
    kpi.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CCCCCC")), ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
                             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F6F7FA"))]))
    story.append(kpi)
    story.append(Spacer(1, 6))
    story.append(Table([[_verdict_chart(docs), _severity_chart(counts)]], colWidths=[W / 2, W / 2]))

    story.append(Paragraph("Resumo por documento", st["H2x"]))
    rows = [[Paragraph("<b>Documento</b>", st["Sx"]), Paragraph("<b>Tipo</b>", st["Sx"]), Paragraph("<b>Págs.</b>", st["Sx"]),
             Paragraph("<b>Achados</b>", st["Sx"]), Paragraph("<b>Conformidade</b>", st["Sx"]), Paragraph("<b>Veredito</b>", st["Sx"])]]
    for d in docs:
        n = sum(1 for f in findings_by_doc.get(d["id"], []) if f.status == "ativo")
        rows.append([Paragraph(rl(d["name"]), st["Sx"]), Paragraph(rl(d["kind"]), st["Sx"]), Paragraph(str(d["n_pages"]), st["Sx"]),
                     Paragraph(str(n), st["Sx"]), Paragraph(f"{d['conformity']:.0f}%", st["Sx"]), _verdict_cell(d["verdict"], st)])
    t = Table(rows, colWidths=[W * 0.36, W * 0.09, W * 0.08, W * 0.1, W * 0.15, W * 0.22], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9ECF3")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(t)

    for d in docs:
        story.append(PageBreak())
        fs = sorted([f for f in findings_by_doc.get(d["id"], [])], key=lambda f: f.num)
        story.append(Paragraph(rl(d["name"]), st["H2x"]))
        head = Table([[Paragraph(f"Tipo: {rl(d['kind'])} | Páginas: {d['n_pages']} | Chunks no banco vetorial: {d['n_chunks']}", st["Sx"]),
                       Paragraph(f"Risco {d['risk']:.2f} | Conformidade {d['conformity']:.0f}%", st["Sx"]), _verdict_cell(d["verdict"], st)]],
                     colWidths=[W * 0.52, W * 0.26, W * 0.22])
        head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC"))]))
        story.append(head)
        story.append(Spacer(1, 4))
        story.append(Paragraph(rl(d["rationale"]), st["Bx"]))
        llm = (d.get("extra") or {}).get("llm")
        if llm and "justificativa" in llm:
            story.append(Paragraph(f"<b>Parecer do juiz LLM:</b> {rl(str(llm['justificativa']))}", st["Sx"]))

        if fs:
            story.append(Paragraph("Achados e localização", st["H3x"]))
            rows = [[Paragraph(f"<b>{h}</b>", st["Sx"]) for h in ("#", "Pág.", "Modal.", "Regra", "Sev.", "Onde", "Trecho")]]
            for f in fs:
                col = SEV_COLOR.get(f.severity, "#555555")
                ev = ("<br/><i>" + rl("; ".join(f.evidence[:3])) + "</i>") if f.evidence else ""
                rows.append([
                    Paragraph(str(f.num), st["Sx"]), Paragraph(str(f.page), st["Sx"]), Paragraph(rl(f.modality), st["Sx"]),
                    Paragraph(rl(f.rule) + ("" if f.status == "ativo" else " (descartado)"), st["Sx"]),
                    Paragraph(f'<b>{SEV_LABEL.get(f.severity, f.severity)}</b><br/>{f.confidence:.2f}', ParagraphStyle("sv", parent=st["Sx"], backColor=colors.HexColor(col), textColor=colors.HexColor("#0b0b0b" if f.severity in ("media", "baixa") else "#ffffff"))),
                    Paragraph(rl(_where(f)) + ("<br/>oculto" if f.hidden else ""), st["Sx"]),
                    Paragraph(rl(f.snippet[:220]) + ev, st["Sx"]),
                ])
            t = Table(rows, colWidths=[W * 0.045, W * 0.05, W * 0.075, W * 0.19, W * 0.075, W * 0.2, W * 0.365], repeatRows=1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9ECF3")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
            story.append(t)

            pages = {}
            for pg in (d.get("pages") or []):
                pages[pg["n"]] = pg
            shown = 0
            for n in sorted({f.page for f in fs if f.bbox and f.status == "ativo"}):
                pg = pages.get(n)
                if not pg or not pg.get("img") or shown >= max_pages_per_doc:
                    continue
                try:
                    img = highlight_page(pg["img"], [f for f in fs if f.page == n], pg["w"])
                except Exception:
                    continue
                img.thumbnail((1100, 1500))
                b = io.BytesIO()
                img.save(b, "JPEG", quality=82)
                b.seek(0)
                h_max = 12.5 * cm
                ratio = img.height / img.width
                w_img = min(W * 0.62, h_max / ratio)
                story.append(KeepTogether([Paragraph(f"Página {n}: áreas destacadas (números correspondem à tabela)", st["H3x"]),
                                           RLImage(b, width=w_img, height=w_img * ratio)]))
                shown += 1
        else:
            story.append(Paragraph("Nenhum achado nas camadas de texto, imagem, metadados e semântica.", st["Bx"]))

        trace = d.get("trace") or []
        if trace:
            story.append(Paragraph("Trilha dos agentes", st["H3x"]))
            rows = [[Paragraph(f"<b>{h}</b>", st["Sx"]) for h in ("Agente", "Ação", "Detalhe", "ms")]]
            for tr in trace:
                rows.append([Paragraph(rl(str(tr["agente"])), st["Sx"]), Paragraph(rl(str(tr["acao"])), st["Sx"]),
                             Paragraph(rl(str(tr["detalhe"])), st["Sx"]), Paragraph(f"{tr['ms']:.0f}", st["Sx"])])
            t = Table(rows, colWidths=[W * 0.16, W * 0.18, W * 0.58, W * 0.08], repeatRows=1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F3F8"))]))
            story.append(t)

    story.append(PageBreak())
    story.append(Paragraph("Metodologia e parâmetros", st["H2x"]))
    rows = [[Paragraph("<b>Camada</b>", st["Sx"]), Paragraph("<b>Função</b>", st["Sx"])]]
    for a, b in LAYERS:
        rows.append([Paragraph(rl(a), st["Sx"]), Paragraph(rl(b), st["Sx"])])
    t = Table(rows, colWidths=[W * 0.28, W * 0.72])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9ECF3")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(Paragraph("Parâmetros em uso", st["H3x"]))
    rows = [[Paragraph(f"<b>{rl(k)}</b>", st["Sx"]), Paragraph(rl(str(v)), st["Sx"])] for k, v in params.items()]
    t = Table(rows, colWidths=[W * 0.35, W * 0.65])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD"))]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(Paragraph("O relatório aponta indícios técnicos e não substitui a revisão humana. Achados com veredito SUSPEITO pedem conferência "
                           "manual do trecho indicado.", st["Sx"]))
    pdf.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
