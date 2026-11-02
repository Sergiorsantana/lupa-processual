"""Camada 1: ingestão multimodal (PDF, DOCX, TXT, imagens) com extração de atributos de renderização."""
from __future__ import annotations

import io
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pymupdf
from PIL import Image

from . import vision
from .config import PAGES_DIR
from .detectors import hidden_groups, luminance, run_reasons
from .models import DocData, PageData, Run, Unit

TEXT_EXT = {".txt", ".md", ".csv", ".json", ".html", ".htm", ".xml"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def kind_of(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in (".docx", ".dotx"):
        return "docx"
    if ext in IMAGE_EXT:
        return "imagem"
    if ext in TEXT_EXT:
        return "txt"
    raise ValueError(f"Formato não suportado: {ext}")


# ---------------------------------------------------------------- PDF

def _color(c) -> tuple:
    if c is None:
        return (0.0, 0.0, 0.0)
    if isinstance(c, (int, float)):
        return (float(c),) * 3
    c = tuple(float(x) for x in c)
    if len(c) == 1:
        return (c[0],) * 3
    if len(c) == 4:  # CMYK
        k = c[3]
        return tuple((1 - min(1.0, v + k)) for v in c[:3])
    return c[:3]


def _merge_runs(runs: list[Run]) -> list[Run]:
    """Junta trechos contíguos da mesma linha e com a mesma causa de ocultação."""
    out: list[Run] = []
    for r in runs:
        if out:
            p = out[-1]
            same_key = sorted(x.split(" (")[0] for x in p.reasons) == sorted(x.split(" (")[0] for x in r.reasons)
            size = max(p.size, r.size, 1.0)
            same_line = abs(p.bbox[1] - r.bbox[1]) <= 0.4 * size and abs(p.bbox[3] - r.bbox[3]) <= 0.6 * size
            gap = r.bbox[0] - p.bbox[2]
            if same_key and same_line and -0.3 * size <= gap <= 1.5 * size:
                sep = "" if gap < 0.12 * size else " "
                p.text = p.text + sep + r.text
                p.bbox = (min(p.bbox[0], r.bbox[0]), min(p.bbox[1], r.bbox[1]), max(p.bbox[2], r.bbox[2]), max(p.bbox[3], r.bbox[3]))
                p.size = min(p.size, r.size) if p.size and r.size else max(p.size, r.size)
                continue
        out.append(r)
    return out


def ingest_pdf(path: str, doc_id: str, cfg, progress=None) -> DocData:
    pdf = pymupdf.open(path)
    dd = DocData(name=Path(path).name, kind="pdf", path=path)
    out_dir = PAGES_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    off_layers: set = set()
    try:
        off_layers = {v["name"] for v in pdf.get_ocgs().values() if not v.get("on", True)}
    except Exception:
        pass
    dd.extra["ocg_off"] = sorted(off_layers)

    zoom = cfg.render_zoom
    ocr_used = 0
    n_img_total = 0
    for i, page in enumerate(pdf):
        if progress:
            progress(f"Lendo página {i + 1} de {len(pdf)}")
        W, H = page.rect.width, page.rect.height
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        png = out_dir / f"p{i + 1}.png"
        pix.save(str(png))
        pd = PageData(number=i + 1, width=W, height=H, image_path=str(png), scale=zoom)

        dark_rects, cover_rects = [], []
        try:
            for d in page.get_drawings():
                fill = d.get("fill")
                if fill is None:
                    continue
                r = d["rect"]
                if (d.get("fill_opacity") or 1.0) < 0.9:
                    continue
                rect = (r.x0, r.y0, r.x1, r.y1)
                if luminance(_color(fill)) < 0.5:
                    dark_rects.append(rect)
                if (r.x1 - r.x0) * (r.y1 - r.y0) < W * H * 0.9:  # ignora o fundo da página inteira
                    cover_rects.append((rect, d.get("seqno", 0)))
        except Exception:
            pass

        image_rects, img_infos = [], []
        try:
            for info in page.get_image_info(xrefs=True):
                b = info["bbox"]
                image_rects.append((b[0], b[1], b[2], b[3]))
                img_infos.append(info)
        except Exception:
            pass

        raw: list[Run] = []
        for t in page.get_texttrace():
            text = "".join(chr(c[0]) for c in t["chars"])
            if not text.strip():
                continue
            raw.append(Run(text=text, bbox=tuple(t["bbox"]), size=float(t.get("size") or 0), color=_color(t.get("color")),
                           opacity=float(t.get("opacity", 1.0)), mode=int(t.get("type", 0)), layer=t.get("layer") or None,
                           seqno=int(t.get("seqno", 0))))
        for r in raw:
            r.reasons = run_reasons(r, W, H, cfg, dark_rects, image_rects, cover_rects, off_layers)
        pd.runs = _merge_runs(raw)
        pd.blocks = [Unit(text=r.text, bbox=r.bbox, page=i + 1) for r in pd.runs if not r.reasons]
        pd.text_layer = "\n".join(r.text for r in pd.runs)

        # OCR da página renderizada: o que o olho humano enxerga
        if cfg.ocr_enabled and vision.ocr_available() and ocr_used < cfg.ocr_max_pages:
            ocr_used += 1
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            lines = vision.ocr_lines(img, cfg, scale=zoom, modality="imagem")
            pd.ocr_all = " ".join(u.text for u in lines)
            layer_tokens = vision.norm_tokens(pd.text_layer)
            for u in lines:
                u.page = i + 1
                toks = vision.norm_tokens(u.text)
                if toks and len(toks - layer_tokens) / len(toks) >= 0.6:
                    pd.ocr_lines.append(u)

        # imagens embutidas
        for info in img_infos:
            if n_img_total >= cfg.max_embedded_images:
                break
            xref = info.get("xref") or 0
            if not xref:
                continue
            try:
                ex = pdf.extract_image(xref)
                pil = Image.open(io.BytesIO(ex["image"]))
                pil.load()
                if min(pil.size) < cfg.min_image_px:
                    continue
                pil = pil.convert("RGB")
            except Exception:
                continue
            n_img_total += 1
            b = info["bbox"]
            pd.images.append({"xref": xref, "bbox": (b[0], b[1], b[2], b[3]), "pil": pil})
        dd.pages.append(pd)

    # metadados, anotações, anexos, scripts
    md = pdf.metadata or {}
    for k, v in md.items():
        if isinstance(v, str) and len(v.strip()) > 3:
            dd.meta_units.append(Unit(text=v, locator=f"metadado PDF: {k}", hidden=True, modality="metadado", page=1))
    try:
        xmp = pdf.get_xml_metadata() or ""
        xmp_text = re.sub(r"<[^>]+>", " ", xmp)
        xmp_text = re.sub(r"\s+", " ", xmp_text).strip()
        if len(xmp_text) > 20:
            dd.meta_units.append(Unit(text=xmp_text, locator="metadado XMP", hidden=True, modality="metadado", page=1))
    except Exception:
        pass
    for pd in dd.pages:
        page = pdf[pd.number - 1]
        try:
            for a in page.annots() or []:
                content = (a.info or {}).get("content", "") or ""
                if len(content.strip()) > 3:
                    flags = a.flags or 0
                    hid = bool(flags & 2 or flags & 32)
                    r = a.rect
                    dd.meta_units.append(Unit(text=content, bbox=(r.x0, r.y0, r.x1, r.y1), hidden=hid, modality="metadado", page=pd.number,
                                              locator=f"anotação {a.type[1]}" + (" oculta" if hid else ""),
                                              reasons=["anotação com flag de oculta"] if hid else []))
        except Exception:
            pass
        try:
            for w in page.widgets() or []:
                val = str(w.field_value or "")
                if len(val.strip()) > 3:
                    r = w.rect
                    dd.meta_units.append(Unit(text=val, bbox=(r.x0, r.y0, r.x1, r.y1), hidden=False, modality="metadado", page=pd.number,
                                              locator=f"campo de formulário {w.field_name}"))
        except Exception:
            pass
    try:
        for name in pdf.embfile_names():
            raw = pdf.embfile_get(name)
            dd.extra.setdefault("embedded", []).append((name, len(raw)))
            if len(raw) <= 300_000:
                s = raw.decode("utf-8", errors="ignore")
                if s and sum(ch.isprintable() or ch in "\n\t\r" for ch in s) / len(s) > 0.9:
                    dd.meta_units.append(Unit(text=s, hidden=True, modality="metadado", page=1, locator=f"arquivo anexado: {name}"))
    except Exception:
        pass
    try:
        for x in range(1, min(pdf.xref_length(), 20000)):
            s = pdf.xref_object(x, compressed=False)
            if "/JavaScript" in s or "/JS" in s or "/Launch" in s:
                dd.extra.setdefault("scripts", []).append((x, re.sub(r"\s+", " ", s)[:200]))
    except Exception:
        pass
    pdf.close()
    return dd


# ---------------------------------------------------------------- DOCX

def _docx_run_reasons(run, cfg) -> list[str]:
    reasons = []
    try:
        if run.font.hidden or (run.style is not None and run.style.font is not None and run.style.font.hidden):
            reasons.append("texto marcado como oculto (vanish)")
    except Exception:
        pass
    try:
        if run.font.size is not None and run.font.size.pt < cfg.min_font_size:
            reasons.append(f"fonte minúscula ({run.font.size.pt:.1f} pt)")
    except Exception:
        pass
    try:
        rgb = run.font.color.rgb if run.font.color is not None else None
        if rgb is not None:
            c = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
            has_shading = run._r.rPr is not None and run._r.rPr.find(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}shd") is not None
            if luminance(c) >= cfg.white_luminance and not has_shading:
                reasons.append("cor igual ao fundo (branco)")
    except Exception:
        pass
    return reasons


def ingest_docx(path: str, doc_id: str, cfg, progress=None) -> DocData:
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = Document(path)
    dd = DocData(name=Path(path).name, kind="docx", path=path)

    def paragraphs():
        pi = 0
        for child in d.element.body.iterchildren():
            if child.tag == qn("w:p"):
                pi += 1
                yield f"parágrafo {pi}", Paragraph(child, d)
            elif child.tag == qn("w:tbl"):
                tbl = Table(child, d)
                for ri, row in enumerate(tbl.rows, 1):
                    for ci, cell in enumerate(row.cells, 1):
                        for p in cell.paragraphs:
                            yield f"tabela, linha {ri}, célula {ci}", p
        seen = set()
        for k, tx in enumerate(d.element.body.iter(qn("w:txbxContent")), 1):
            for p in tx.iter(qn("w:p")):
                par = Paragraph(p, d)
                if par.text.strip() and par.text not in seen:
                    seen.add(par.text)
                    yield f"caixa de texto {k}", par
        for si, sec in enumerate(d.sections, 1):
            for lab, part in (("cabeçalho", sec.header), ("rodapé", sec.footer)):
                for p in part.paragraphs:
                    if p.text.strip():
                        yield f"{lab} (seção {si})", p

    for loc, par in paragraphs():
        cur_text, cur_key, cur_reasons = [], None, []

        def flush():
            nonlocal cur_text, cur_key, cur_reasons
            if cur_text and "".join(cur_text).strip():
                hid = bool(cur_reasons)
                dd.paragraph_units.append(Unit(text="".join(cur_text), locator=loc + (" (trecho oculto)" if hid else ""),
                                               hidden=hid, reasons=list(cur_reasons), page=1))
            cur_text, cur_key, cur_reasons = [], None, []

        for run in par.runs:
            rs = _docx_run_reasons(run, cfg)
            key = tuple(sorted(x.split(" (")[0] for x in rs))
            if cur_text and key != cur_key:
                flush()
            cur_text.append(run.text)
            cur_key, cur_reasons = key, rs
        flush()

    cp = d.core_properties
    for k in ("title", "subject", "author", "keywords", "comments", "category", "last_modified_by"):
        v = getattr(cp, k, None)
        if isinstance(v, str) and len(v.strip()) > 3:
            dd.meta_units.append(Unit(text=v, locator=f"propriedade do documento: {k}", hidden=True, modality="metadado"))
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if "word/comments.xml" in names:
                xml = z.read("word/comments.xml").decode("utf-8", errors="ignore")
                txt = " ".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml, flags=re.S))
                if txt.strip():
                    dd.meta_units.append(Unit(text=txt, locator="comentários do documento", hidden=False, modality="metadado"))
            if "docProps/custom.xml" in names:
                xml = z.read("docProps/custom.xml").decode("utf-8", errors="ignore")
                txt = " ".join(re.findall(r"<vt:lpwstr>(.*?)</vt:lpwstr>", xml, flags=re.S))
                if txt.strip():
                    dd.meta_units.append(Unit(text=txt, locator="propriedades personalizadas", hidden=True, modality="metadado"))
            if "word/document.xml" in names:
                xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
                dels = " ".join(re.findall(r"<w:delText[^>]*>(.*?)</w:delText>", xml, flags=re.S))
                if dels.strip():
                    dd.meta_units.append(Unit(text=dels, locator="texto excluído em controle de alterações", hidden=True,
                                              reasons=["texto excluído ainda presente no arquivo"], modality="metadado"))
    except Exception:
        pass

    # Renderização opcional por LibreOffice, para localizar visualmente os achados
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", tmp, path],
                               capture_output=True, timeout=120, check=True)
                pdfs = list(Path(tmp).glob("*.pdf"))
                if pdfs:
                    out_dir = PAGES_DIR / doc_id
                    out_dir.mkdir(parents=True, exist_ok=True)
                    keep = out_dir / "render.pdf"
                    shutil.copy(pdfs[0], keep)
                    pdf = pymupdf.open(str(keep))
                    for i, page in enumerate(pdf):
                        pix = page.get_pixmap(matrix=pymupdf.Matrix(cfg.render_zoom, cfg.render_zoom), alpha=False)
                        png = out_dir / f"p{i + 1}.png"
                        pix.save(str(png))
                        dd.pages.append(PageData(number=i + 1, width=page.rect.width, height=page.rect.height,
                                                 image_path=str(png), scale=cfg.render_zoom))
                    pdf.close()
                    dd.extra["render_pdf"] = str(keep)
        except Exception:
            pass
    if not dd.pages:
        dd.pages.append(PageData(number=1, width=0, height=0))
    return dd


def locate_in_render(dd: DocData, snippet: str) -> tuple[int, tuple] | None:
    """Localiza um trecho do DOCX no PDF renderizado (apenas texto que aparece na página)."""
    path = dd.extra.get("render_pdf")
    if not path:
        return None
    needle = re.sub(r"\s+", " ", snippet).strip()[:50]
    if len(needle) < 8:
        return None
    pdf = pymupdf.open(path)
    try:
        for i, page in enumerate(pdf):
            hits = page.search_for(needle)
            if hits:
                r = hits[0]
                return i + 1, (r.x0, r.y0, r.x1, r.y1)
    finally:
        pdf.close()
    return None


# ---------------------------------------------------------------- TXT

def ingest_txt(path: str, doc_id: str, cfg, progress=None) -> DocData:
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    dd = DocData(name=Path(path).name, kind="txt", path=path)
    for n, line in enumerate(text.split("\n"), 1):
        if line.strip():
            dd.paragraph_units.append(Unit(text=line, locator=f"linha {n}", page=1))
    dd.pages.append(PageData(number=1, width=0, height=0, text_layer=text))
    return dd


# ---------------------------------------------------------------- imagem

def ingest_image(path: str, doc_id: str, cfg, progress=None) -> DocData:
    dd = DocData(name=Path(path).name, kind="imagem", path=path)
    orig = Image.open(path)
    orig.load()
    meta = vision.image_text_metadata(orig)
    rgb = orig.convert("RGB")
    W, H = rgb.size
    out_dir = PAGES_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    disp = rgb
    if max(W, H) > 2200:
        r = 2200 / max(W, H)
        disp = rgb.resize((int(W * r), int(H * r)))
    png = out_dir / "p1.png"
    disp.save(png)
    scale = disp.width / W
    pd = PageData(number=1, width=float(W), height=float(H), image_path=str(png), scale=scale)
    if cfg.ocr_enabled and vision.ocr_available():
        lines = vision.ocr_lines(disp, cfg, scale=scale, modality="imagem")
        for u in lines:
            u.page = 1
        pd.ocr_lines = lines
        pd.ocr_all = " ".join(u.text for u in lines)
    pd.images.append({"xref": 0, "bbox": (0.0, 0.0, float(W), float(H)), "pil": rgb, "disp": disp, "disp_scale": scale})
    dd.pages.append(pd)
    for k, v in meta:
        dd.meta_units.append(Unit(text=v, locator=f"metadado da imagem: {k}", hidden=True, modality="metadado", page=1))
    tail = vision.trailing_data(path)
    if len(tail) > 16:
        dd.extra["trailing"] = tail
        s = tail.decode("utf-8", errors="ignore")
        if s and sum(ch.isprintable() or ch in "\n\t\r" for ch in s) / len(s) > 0.85:
            dd.meta_units.append(Unit(text=s, locator="dados anexados após o fim do arquivo", hidden=True, modality="metadado",
                                      reasons=["bytes depois do marcador de fim da imagem"], page=1))
    return dd


def ingest(path: str, doc_id: str, cfg, progress=None) -> DocData:
    k = kind_of(path)
    fn = {"pdf": ingest_pdf, "docx": ingest_docx, "txt": ingest_txt, "imagem": ingest_image}[k]
    return fn(path, doc_id, cfg, progress)
