"""Camada 1/2 visual: OCR, realce de baixo contraste, esteganografia LSB e dados anexados."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from PIL import Image, ImageFilter

from .models import Unit

try:
    import pytesseract
    from pytesseract import Output
    _HAS_TESS = True
except Exception:  # pragma: no cover
    _HAS_TESS = False


@lru_cache(maxsize=1)
def _langs() -> set:
    try:
        return set(pytesseract.get_languages())
    except Exception:
        return set()


def ocr_available() -> bool:
    return _HAS_TESS and bool(_langs())


def resolve_lang(lang: str) -> str:
    have = _langs()
    parts = [p for p in lang.split("+") if p in have]
    return "+".join(parts) or ("eng" if "eng" in have else "")


def ocr_lines(img: Image.Image, cfg, scale: float = 1.0, origin: tuple = (0.0, 0.0), modality: str = "imagem") -> list[Unit]:
    """Executa OCR e devolve linhas com bbox convertido para unidades da página (px / scale + origem)."""
    if not ocr_available():
        return []
    lang = resolve_lang(cfg.ocr_lang)
    try:
        d = pytesseract.image_to_data(img, lang=lang, output_type=Output.DICT, config="--psm 3")
    except Exception:
        return []
    lines: dict = {}
    for i, txt in enumerate(d["text"]):
        txt = (txt or "").strip()
        try:
            conf = float(d["conf"][i])
        except Exception:
            conf = -1
        if not txt or conf < cfg.ocr_min_conf:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        ln = lines.setdefault(key, {"words": [], "x0": x, "y0": y, "x1": x + w, "y1": y + h})
        ln["words"].append(txt)
        ln["x0"], ln["y0"] = min(ln["x0"], x), min(ln["y0"], y)
        ln["x1"], ln["y1"] = max(ln["x1"], x + w), max(ln["y1"], y + h)
    out = []
    ox, oy = origin
    for key in sorted(lines):
        ln = lines[key]
        bbox = (ox + ln["x0"] / scale, oy + ln["y0"] / scale, ox + ln["x1"] / scale, oy + ln["y1"] / scale)
        out.append(Unit(text=" ".join(ln["words"]), bbox=bbox, modality=modality))
    return out


def amplify(img: Image.Image, low: float, high: float, window: int) -> Image.Image:
    """Isola diferenças pequenas de tom em relação ao fundo local (texto quase da cor do fundo).

    Pixels que diferem do fundo (mediana local) entre `low` e `high` níveis viram preto. Conteúdo visível (diferença
    acima de `high`) e a borda ao redor dele são descartados, para o OCR enxergar só o que o olho não vê.
    """
    g = img.convert("L")
    if max(g.size) > 1800:
        r = 1800 / max(g.size)
        g = g.resize((max(int(g.width * r), 1), max(int(g.height * r), 1)))
    w = window if window % 2 == 1 else window + 1
    bg = g.filter(ImageFilter.MedianFilter(w))
    d = np.abs(np.asarray(g, dtype=np.float32) - np.asarray(bg, dtype=np.float32))
    strong = Image.fromarray(((d > high) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7))
    near = np.asarray(strong) > 0
    mask = (d >= low) & (d <= high) & ~near
    m = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))
    out = 255 - np.asarray(m)
    return Image.fromarray(out.astype(np.uint8))


def norm_tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]{4,}", text.lower())}


def new_text_in_amplified(img: Image.Image, cfg, scale: float = 1.0, origin=(0.0, 0.0), normal: list | None = None) -> list[Unit]:
    """OCR da imagem realçada, mantendo só linhas que o OCR normal não enxerga."""
    if not ocr_available():
        return []
    amp = amplify(img, cfg.amplify_low, cfg.amplify_high, cfg.amplify_window)
    factor = img.width / amp.width if amp.width else 1.0
    if normal is None:
        normal = ocr_lines(img, cfg, scale=scale, origin=origin)
    seen = norm_tokens(" ".join(u.text for u in normal))
    lines = ocr_lines(amp, cfg, scale=scale / factor if factor else scale, origin=origin)
    res = []
    for u in lines:
        toks = norm_tokens(u.text)
        letters = sum(ch.isalpha() for ch in u.text)
        if len(u.text) < 10 or letters / max(len(u.text), 1) < 0.6 or not toks:
            continue
        if len(toks - seen) / len(toks) >= 0.6:
            res.append(u)
    return res


def lsb_message(img: Image.Image, min_chars: int) -> str:
    """Tenta extrair texto ASCII legível dos bits menos significativos (esteganografia LSB simples)."""
    try:
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    except Exception:
        return ""
    candidates = [arr.reshape(-1), arr[:, :, 0].reshape(-1)]
    rx = re.compile(rb"[\x20-\x7e\r\n\t]{%d,}" % min_chars)
    for flat in candidates:
        bits = (flat[: 8 * 6000] & 1).astype(np.uint8)
        n = (len(bits) // 8) * 8
        if n == 0:
            continue
        by = np.packbits(bits[:n]).tobytes()
        m = rx.search(by)
        if m:
            return m.group(0).decode("latin-1").strip()
    return ""


def trailing_data(path: str) -> bytes:
    """Bytes adicionados depois do marcador de fim de PNG ou JPEG."""
    try:
        raw = open(path, "rb").read()
    except Exception:
        return b""
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        i = raw.rfind(b"IEND")
        return raw[i + 8:] if i >= 0 else b""
    if raw[:2] == b"\xff\xd8":
        i = raw.rfind(b"\xff\xd9")
        return raw[i + 2:] if i >= 0 else b""
    return b""


def image_text_metadata(img: Image.Image) -> list[tuple[str, str]]:
    """Campos de texto de metadados: chunks PNG, comentário JPEG, EXIF e XMP."""
    out: list[tuple[str, str]] = []
    for k, v in (img.info or {}).items():
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="ignore")
        if isinstance(v, str) and len(v) > 3 and k not in ("icc_profile",):
            out.append((str(k), v))
    try:
        ex = img.getexif()
        for tag, v in ex.items():
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="ignore")
            if isinstance(v, str) and len(v) > 3:
                out.append((f"EXIF {tag}", v))
    except Exception:
        pass
    return out
