"""Camada 3: construção de chunks multimodais, indexação e recuperação (RAG)."""
from __future__ import annotations

import json
import uuid

from .detectors import hidden_groups, normalize_for_match
from .models import DocData, Unit


def _union(units: list[Unit]):
    bb = [u.bbox for u in units if u.bbox]
    if not bb:
        return None
    return (min(b[0] for b in bb), min(b[1] for b in bb), max(b[2] for b in bb), max(b[3] for b in bb))


def chunk_units(units: list[Unit], size: int, overlap: int, *, hidden: bool, modality: str, page: int) -> list[dict]:
    chunks: list[dict] = []
    cur: list[Unit] = []
    cur_len = 0

    def emit(text, bbox, loc):
        text = normalize_for_match(text)
        if len(text) >= 3:
            chunks.append({"text": text, "bbox": bbox, "locator": loc, "page": page, "hidden": hidden, "modality": modality})

    def flush():
        if cur:
            emit(" ".join(u.text for u in cur), _union(cur), cur[0].locator)

    for u in units:
        t = u.text.strip()
        if not t:
            continue
        if len(t) > size:
            flush()
            cur, cur_len = [], 0
            step = max(size - overlap, 50)
            for s in range(0, len(t), step):
                emit(t[s:s + size], u.bbox, u.locator)
                if s + size >= len(t):
                    break
            continue
        if cur and cur_len + len(t) + 1 > size:
            flush()
            keep, l = [], 0
            for x in reversed(cur):
                if l + len(x.text) > overlap:
                    break
                keep.insert(0, x)
                l += len(x.text)
            cur, cur_len = keep, sum(len(x.text) + 1 for x in keep)
        cur.append(u)
        cur_len += len(t) + 1
    flush()
    return chunks


def build_chunks(dd: DocData, cfg) -> list[dict]:
    size, ov = cfg.chunk_size, cfg.chunk_overlap
    out: list[dict] = []
    for pd in dd.pages:
        # texto visível da camada de texto
        out += chunk_units(pd.blocks, size, ov, hidden=False, modality="texto", page=pd.number)
        # texto oculto (formatação)
        for g in hidden_groups(pd.runs):
            units = [Unit(text=r.text, bbox=r.bbox, locator="texto oculto") for r in g]
            out += chunk_units(units, size, ov, hidden=True, modality="texto", page=pd.number)
        # texto lido por OCR que não está na camada de texto (imagens, páginas digitalizadas)
        out += chunk_units(pd.ocr_lines, size, ov, hidden=False, modality="imagem", page=pd.number)
    vis = [u for u in dd.paragraph_units if not u.hidden]
    hid = [u for u in dd.paragraph_units if u.hidden]
    out += chunk_units(vis, size, ov, hidden=False, modality="texto", page=1)
    out += chunk_units(hid, size, ov, hidden=True, modality="texto", page=1)
    for u in dd.meta_units:
        out += chunk_units([u], size, ov, hidden=u.hidden, modality="metadado", page=u.page)
    return out


def index_document(store, embedder, dd: DocData, doc_id: str, cfg) -> int:
    chunks = build_chunks(dd, cfg)
    if not chunks:
        return 0
    emb = embedder.embed([c["text"] for c in chunks])
    ids = [f"{doc_id}-{i}-{uuid.uuid4().hex[:6]}" for i in range(len(chunks))]
    metas = [{
        "doc_id": doc_id, "doc_name": dd.name, "page": int(c["page"]), "modality": c["modality"],
        "hidden": bool(c["hidden"]), "bbox": json.dumps(c["bbox"]) if c["bbox"] else "",
        "locator": c["locator"] or "",
    } for c in chunks]
    store.add(ids, emb, [c["text"] for c in chunks], metas)
    return len(chunks)


def probe_document(store, embedder, doc_id: str, probes: list[str], cfg) -> list[dict]:
    """Consulta o banco vetorial com sondas de comandos conhecidos e devolve os trechos mais próximos."""
    if not probes:
        return []
    qs = embedder.embed(probes)
    hits: dict = {}
    for probe, q in zip(probes, qs):
        for h in store.query(q, cfg.top_k, doc_id=doc_id):
            if h["sim"] < cfg.semantic_threshold:
                continue
            key = (h["meta"]["page"], h["meta"].get("bbox", ""), h["text"][:60])
            if key not in hits or h["sim"] > hits[key]["sim"]:
                hits[key] = {**h, "probe": probe}
    return sorted(hits.values(), key=lambda x: -x["sim"])
