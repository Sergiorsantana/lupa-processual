"""Camada 2: detectores determinísticos (as ferramentas que os agentes chamam)."""
from __future__ import annotations

import base64
import binascii
import re
import unicodedata

from .models import Run, Unit

# ---------------------------------------------------------------- caracteres invisíveis

ZERO_WIDTH = {0x200B, 0x200C, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064, 0xFEFF, 0x180E, 0x00AD}
BIDI = set(range(0x202A, 0x202F)) | set(range(0x2066, 0x206A)) | {0x200E, 0x200F, 0x061C}
TAG_RANGE = range(0xE0000, 0xE0080)

CONFUSABLES = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s", "ӏ": "l",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
    "ο": "o", "Ν": "N", "Ο": "O", "Τ": "T", "Ι": "I", "ν": "v", "ρ": "p",
}
_CONF_TABLE = {ord(k): v for k, v in CONFUSABLES.items()}
_STRIP_TABLE = {c: None for c in ZERO_WIDTH | BIDI | set(TAG_RANGE)}


def normalize_for_match(text: str) -> str:
    """Normaliza o texto para o regex: remove invisíveis, desfaz homóglifos e compatibilidades Unicode."""
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_STRIP_TABLE).translate(_CONF_TABLE)
    return re.sub(r"\s+", " ", t).strip()


def show_invisibles(text: str) -> str:
    """Torna visíveis os caracteres invisíveis para exibição em tela e relatório."""
    out = []
    for ch in text:
        cp = ord(ch)
        if cp in ZERO_WIDTH or cp in BIDI or cp in TAG_RANGE:
            out.append(f"<U+{cp:04X}>")
        else:
            out.append(ch)
    return "".join(out)


def analyze_invisible(text: str) -> dict:
    zw = [c for c in text if ord(c) in ZERO_WIDTH and ord(c) != 0x00AD]
    if text.startswith("﻿"):
        zw = zw[1:]
    bidi = [c for c in text if ord(c) in BIDI]
    tags = [c for c in text if ord(c) in TAG_RANGE]
    payload_tags = "".join(chr(ord(c) - 0xE0000) for c in tags if 0x20 <= ord(c) - 0xE0000 < 0x7F)
    bits = "".join("0" if c == "​" else "1" for c in text if c in ("​", "‌"))
    payload_bits = ""
    if len(bits) >= 16:
        by = bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits) - len(bits) % 8, 8))
        s = by.decode("latin-1")
        if s and sum(32 <= ord(ch) < 127 for ch in s) / len(s) >= 0.9:
            payload_bits = s
    return {"zw": len(zw), "bidi": len(bidi), "tags": len(tags), "payload_tags": payload_tags, "payload_bits": payload_bits}


def mixed_script_words(text: str, limit: int = 5) -> list[str]:
    out = []
    for w in re.findall(r"\w{3,}", text):
        scripts = set()
        for ch in w:
            if ch.isalpha():
                n = unicodedata.name(ch, "")
                scripts.add(n.split(" ")[0])
        if "LATIN" in scripts and (scripts & {"CYRILLIC", "GREEK"}):
            out.append(w)
            if len(out) >= limit:
                break
    return out


_B64 = re.compile(r"(?:[A-Za-z0-9+/]{4}){10,}(?:==|=)?")


def decode_base64_blobs(text: str) -> list[str]:
    res = []
    for m in _B64.finditer(text):
        try:
            raw = base64.b64decode(m.group(0), validate=True)
        except (binascii.Error, ValueError):
            continue
        s = raw.decode("utf-8", errors="ignore")
        if len(s) >= 16 and sum(ch.isprintable() or ch in "\n\t" for ch in s) / len(s) >= 0.92:
            res.append(s)
    return res


# ---------------------------------------------------------------- padrões (regras)

SEV_RANK = {"critica": 4, "alta": 3, "media": 2, "baixa": 1}


def compile_patterns(patterns: list[dict]) -> list[tuple[dict, re.Pattern]]:
    out = []
    for p in patterns:
        if not p.get("ativo", True):
            continue
        try:
            out.append((p, re.compile(p["regex"], re.I | re.S)))
        except re.error:
            continue
    return out


def _union(bboxes: list) -> tuple | None:
    bb = [b for b in bboxes if b]
    if not bb:
        return None
    return (min(b[0] for b in bb), min(b[1] for b in bb), max(b[2] for b in bb), max(b[3] for b in bb))


def match_units(units: list[Unit], cpats, allow_only_hidden: bool) -> list[dict]:
    """Aplica os padrões sobre uma sequência de unidades e mapeia cada acerto de volta às unidades e ao bbox."""
    if not units:
        return []
    parts, offs, pos = [], [], 0
    for u in units:
        t = normalize_for_match(u.text)
        offs.append((pos, pos + len(t), u))
        parts.append(t)
        pos += len(t) + 1
    full = " ".join(parts)
    cands = []
    for p, rx in cpats:
        if p.get("so_oculto") and not allow_only_hidden:
            continue
        for m in rx.finditer(full):
            if m.end() - m.start() < 4:
                continue
            cands.append((p, m))
    cands.sort(key=lambda pm: (-SEV_RANK.get(pm[0]["severidade"], 1), -(pm[1].end() - pm[1].start())))
    taken: list[tuple[int, int]] = []
    out = []
    for p, m in cands:
        if any(m.start() < b and m.end() > a for a, b in taken):
            continue
        taken.append((m.start(), m.end()))
        involved = [u for (a, b, u) in offs if a < m.end() and b > m.start()]
        out.append({
            "pattern": p,
            "match": m.group(0),
            "snippet": full[max(0, m.start() - 40): m.end() + 140].strip(),
            "units": involved,
            "bbox": _union([u.bbox for u in involved]),
        })
    return out


# ---------------------------------------------------------------- classificação de trechos ocultos (PDF)

def luminance(c: tuple) -> float:
    r, g, b = (list(c) + [0, 0, 0])[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contains(outer, inner, tol: float = 1.0) -> bool:
    return outer[0] - tol <= inner[0] and outer[1] - tol <= inner[1] and outer[2] + tol >= inner[2] and outer[3] + tol >= inner[3]


def _overlap_ratio(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    return (ix * iy) / area


def run_reasons(run: Run, W: float, H: float, cfg, dark_rects, image_rects, cover_rects, off_layers) -> list[str]:
    if not run.text.strip():
        return []
    reasons: list[str] = []
    if run.mode == 3:
        reasons.append("modo de renderização invisível (Tr 3)")
    if run.opacity < cfg.min_opacity:
        reasons.append(f"transparência ({run.opacity:.0%})")
    if 0 < run.size < cfg.min_font_size:
        reasons.append(f"fonte minúscula ({run.size:.1f} pt)")
    x0, y0, x1, y1 = run.bbox
    m = cfg.offpage_margin
    if x1 < -m or y1 < -m or x0 > W + m or y0 > H + m:
        reasons.append("fora da área da página")
    if run.layer and run.layer in off_layers:
        reasons.append(f"camada OCG desativada ({run.layer})")
    if luminance(run.color) >= cfg.white_luminance and run.mode != 3:
        over_dark = any(_contains(r, run.bbox) for r in dark_rects)
        over_img = any(_overlap_ratio(run.bbox, r) > 0.5 for r in image_rects)
        if not over_dark and not over_img:
            reasons.append("cor igual ao fundo (branco)")
    for rect, seq in cover_rects:
        if seq > run.seqno and _contains(rect, run.bbox, tol=1.5):
            reasons.append("coberto por forma opaca")
            break
    return reasons


def hidden_groups(runs: list[Run]) -> list[list[Run]]:
    """Agrupa trechos ocultos consecutivos (ordem do fluxo de conteúdo) que compartilham a mesma causa."""
    groups, cur, cur_key = [], [], None
    for r in runs:
        if not r.reasons:
            if cur:
                groups.append(cur)
                cur, cur_key = [], None
            continue
        key = tuple(sorted(x.split(" (")[0] for x in r.reasons))
        if cur and key != cur_key:
            groups.append(cur)
            cur = []
        cur.append(r)
        cur_key = key
    if cur:
        groups.append(cur)
    return groups


def run_to_unit(r: Run) -> Unit:
    return Unit(text=r.text, bbox=tuple(r.bbox), hidden=True, reasons=list(r.reasons))
