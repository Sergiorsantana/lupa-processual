"""Orquestra as 5 camadas para um arquivo: ingestão, indexação vetorial, agentes, juiz e persistência."""
from __future__ import annotations

import hashlib
import shutil
import time
import uuid
from pathlib import Path

from .agents import Ctx, run_agents
from .config import UPLOADS_DIR
from .detectors import compile_patterns
from .ingest import ingest, kind_of, locate_in_render
from .judge import run_judge
from .rag import index_document


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def process_file(path: str, cfg, patterns: list, probes: list, store, embedder, db, progress=None, name: str | None = None) -> dict:
    def say(msg):
        if progress:
            progress(msg)

    name = name or Path(path).name
    kind_of(path)  # valida a extensão cedo
    sha = sha256(path)
    old = db.find_by_sha(sha)
    doc_id = old["id"] if old else uuid.uuid4().hex[:12]
    if old:  # reindexa o mesmo arquivo com os parâmetros atuais
        store.delete_doc(doc_id)
        db.delete_document(doc_id)

    keep = UPLOADS_DIR / f"{doc_id}{Path(path).suffix.lower()}"
    if Path(path).resolve() != keep.resolve():
        shutil.copy(path, keep)

    t0 = time.time()
    say("Camada 1: ingestão multimodal")
    dd = ingest(str(keep), doc_id, cfg, say)
    dd.name = name
    t_ing = (time.time() - t0) * 1000

    say("Camada 3: indexação no banco vetorial")
    t1 = time.time()
    n_chunks = index_document(store, embedder, dd, doc_id, cfg)
    t_idx = (time.time() - t1) * 1000

    say("Camada 4: agentes em execução")
    ctx = Ctx(doc_id=doc_id, dd=dd, cfg=cfg, cpats=compile_patterns(patterns), probes=probes, store=store, embedder=embedder)
    ctx.trace.insert(0, {"agente": "ingestão", "acao": "leu o arquivo", "detalhe": f"{len(dd.pages)} página(s), tipo {dd.kind}", "ms": round(t_ing, 1)})
    ctx.trace.insert(1, {"agente": "banco vetorial", "acao": "indexou", "detalhe": f"{n_chunks} chunks em {store.kind}:{store.name}", "ms": round(t_idx, 1)})
    findings = run_agents(ctx)

    # DOCX: tenta localizar visualmente os achados no PDF renderizado
    if dd.kind == "docx" and dd.extra.get("render_pdf"):
        for f in findings:
            if f.bbox is None and f.modality == "texto" and not f.hidden:
                loc = locate_in_render(dd, f.match or f.snippet)
                if loc:
                    f.page, bb = loc
                    f.bbox = [round(x, 1) for x in bb]
                    pd = dd.pages[f.page - 1]
                    f.page_w, f.page_h = pd.width, pd.height

    say("Camada 5: juiz")
    result = run_judge(findings, cfg, name)
    ctx.log("juiz", "veredito", f"{result['verdict']} (risco {result['risk']:.2f})")
    if result.get("llm"):
        ctx.log("juiz", "parecer LLM", str(result["llm"])[:200])

    pages = [{"n": p.number, "w": p.width, "h": p.height, "img": p.image_path, "scale": p.scale} for p in dd.pages]
    doc = {"id": doc_id, "name": name, "kind": dd.kind, "sha": sha, "n_pages": len(dd.pages), "n_chunks": n_chunks,
           "indexed_at": time.time(), "verdict": result["verdict"], "risk": result["risk"], "conformity": result["conformity"],
           "rationale": result["rationale"], "trace": ctx.trace, "params": cfg.key_params(), "pages": pages,
           "extra": {"llm": result.get("llm"), "vector_store": f"{store.kind}:{store.name}"}, "source_path": str(keep)}
    db.save_document(doc, findings)
    db.log("info", f"{name}: {result['verdict']} com {len(findings)} achados")
    say("Concluído")
    return {"doc": doc, "findings": findings}
