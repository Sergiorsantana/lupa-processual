"""Camada 4: agentes especializados e orquestrador (plano, execução, verificação cruzada)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from . import vision
from .detectors import (analyze_invisible, decode_base64_blobs, hidden_groups, match_units, mixed_script_words,
                        run_to_unit, show_invisibles)
from .models import DocData, Finding, Unit
from .rag import probe_document

SEV_UP = {"baixa": "media", "media": "alta", "alta": "critica", "critica": "critica"}


@dataclass
class Ctx:
    doc_id: str
    dd: DocData
    cfg: object
    cpats: list
    probes: list
    store: object
    embedder: object
    findings: list = field(default_factory=list)
    trace: list = field(default_factory=list)

    def log(self, agente: str, acao: str, detalhe: str = "", ms: float = 0.0):
        self.trace.append({"agente": agente, "acao": acao, "detalhe": detalhe, "ms": round(ms, 1)})

    def page(self, n: int):
        return self.dd.pages[n - 1] if 0 < n <= len(self.dd.pages) else None

    def add(self, agent: str, page: int, modality: str, rule: str, category: str, severity: str, hidden: bool, snippet: str,
            bbox=None, locator: str = "", evidence=None, match: str = "") -> Finding:
        pd = self.page(page)
        f = Finding(doc_id=self.doc_id, doc_name=self.dd.name, page=page, modality=modality, agent=agent, rule=rule,
                    category=category, severity=severity, hidden=hidden, snippet=snippet.strip()[:400],
                    bbox=[round(float(x), 1) for x in bbox] if bbox else None,
                    page_w=pd.width if pd else 0.0, page_h=pd.height if pd else 0.0, locator=locator,
                    evidence=list(evidence or []), match=match.strip()[:200])
        self.findings.append(f)
        return f


# ---------------------------------------------------------------- utilidades compartilhadas

def _pattern_findings(ctx: Ctx, agent: str, units: list[Unit], page: int, modality: str, hidden: bool,
                      allow_only_hidden: bool, extra_evidence=None, locator: str = ""):
    n = 0
    for m in match_units(units, ctx.cpats, allow_only_hidden):
        p = m["pattern"]
        sev = SEV_UP[p["severidade"]] if hidden else p["severidade"]
        reasons = []
        for u in m["units"]:
            reasons += u.reasons
        ev = list(dict.fromkeys(reasons)) + list(extra_evidence or [])
        loc = locator or (m["units"][0].locator if m["units"] else "")
        ctx.add(agent, page, modality, p["nome"], p["categoria"], sev, hidden, m["snippet"], m["bbox"], loc, ev, match=m["match"])
        n += 1
    return n


def _unicode_findings(ctx: Ctx, agent: str, u: Unit, page: int, modality: str, hidden: bool):
    a = analyze_invisible(u.text)
    loc = u.locator
    for key, label in (("payload_tags", "caracteres de tag Unicode"), ("payload_bits", "caracteres de largura zero (bits)")):
        payload = a[key]
        if payload and len(payload) >= 8:
            hit = match_units([Unit(text=payload)], ctx.cpats, True)
            sev = "critica" if hit else "alta"
            ctx.add(agent, page, modality, f"Texto codificado em {label}", "Conteúdo escondido", sev, True,
                    payload, u.bbox, loc, [f"conteúdo decodificado: {payload[:120]}"])
    if a["zw"] >= 3 or a["bidi"] >= 1:
        ctx.add(agent, page, modality, "Caracteres Unicode invisíveis", "Conteúdo escondido", "media", True,
                show_invisibles(u.text)[:300], u.bbox, loc,
                [f"{a['zw']} de largura zero, {a['bidi']} de controle bidirecional"])
    words = mixed_script_words(u.text)
    if words:
        ctx.add(agent, page, modality, "Homóglifos (alfabetos misturados)", "Conteúdo escondido", "media", False,
                ", ".join(words), u.bbox, loc, ["letras de alfabetos diferentes na mesma palavra"])
    for dec in decode_base64_blobs(u.text):
        hit = match_units([Unit(text=dec)], ctx.cpats, True)
        if hit:
            ctx.add(agent, page, modality, "Comando codificado em Base64", "Conteúdo escondido", "alta", True,
                    dec[:300], u.bbox, loc, ["texto Base64 decodificado contém padrão de comando"])


# ---------------------------------------------------------------- agentes

def agent_texto(ctx: Ctx):
    dd, cfg = ctx.dd, ctx.cfg
    for pd in dd.pages:
        for g in hidden_groups(pd.runs):
            units = [run_to_unit(r) for r in g]
            text = " ".join(u.text for u in units)
            n = _pattern_findings(ctx, "texto", units, pd.number, "texto", True, True)
            if n == 0 and len(text.strip()) >= cfg.min_hidden_chars:
                bb = (min(u.bbox[0] for u in units), min(u.bbox[1] for u in units),
                      max(u.bbox[2] for u in units), max(u.bbox[3] for u in units))
                reasons = list(dict.fromkeys(x for u in units for x in u.reasons))
                ctx.add("texto", pd.number, "texto", "Texto oculto sem comando reconhecido", "Conteúdo escondido", "media",
                        True, text, bb, "texto oculto", reasons)
            for u in units:
                _unicode_findings(ctx, "texto", u, pd.number, "texto", True)
        _pattern_findings(ctx, "texto", pd.blocks, pd.number, "texto", False, False)
        for u in pd.blocks:
            _unicode_findings(ctx, "texto", u, pd.number, "texto", False)
    vis = [u for u in dd.paragraph_units if not u.hidden]
    if vis:
        for m in match_units(vis, ctx.cpats, False):
            p = m["pattern"]
            ctx.add("texto", 1, "texto", p["nome"], p["categoria"], p["severidade"], False, m["snippet"], None,
                    m["units"][0].locator if m["units"] else "", [], match=m["match"])
    for u in dd.paragraph_units:
        if u.hidden:
            n = _pattern_findings(ctx, "texto", [u], u.page, "texto", True, True, locator=u.locator)
            if n == 0 and len(u.text.strip()) >= cfg.min_hidden_chars:
                ctx.add("texto", u.page, "texto", "Texto oculto sem comando reconhecido", "Conteúdo escondido", "media", True,
                        u.text, None, u.locator, list(u.reasons))
        _unicode_findings(ctx, "texto", u, u.page, "texto", u.hidden)


def agent_visual(ctx: Ctx):
    dd, cfg = ctx.dd, ctx.cfg
    for pd in dd.pages:
        # verificação cruzada com OCR: o texto oculto aparece na página renderizada?
        if pd.ocr_all:
            seen = vision.norm_tokens(pd.ocr_all)
            for f in ctx.findings:
                if f.agent == "texto" and f.hidden and f.page == pd.number and f.modality == "texto":
                    toks = vision.norm_tokens(f.snippet)
                    if toks:
                        ratio = len(toks & seen) / len(toks)
                        if ratio < 0.3:
                            f.evidence.append("OCR confirmou: o texto não aparece na página renderizada")
                        elif ratio > 0.7:
                            f.evidence.append("Atenção: o OCR também leu este texto, ele pode estar legível")
        # texto presente só na imagem
        if pd.ocr_lines:
            _pattern_findings(ctx, "visual", pd.ocr_lines, pd.number, "imagem", False, False,
                              extra_evidence=["texto lido por OCR, ausente da camada de texto"])
        # imagens embutidas
        for im in pd.images:
            pil, b = im["pil"], im["bbox"]
            if dd.kind == "pdf":
                width_pt = max(b[2] - b[0], 1e-3)
                scale, origin = pil.width / width_pt, (b[0], b[1])
                normal = None
            else:
                scale, origin = 1.0, (0.0, 0.0)
                normal = pd.ocr_lines if pd.ocr_lines else None
            if cfg.ocr_enabled and vision.ocr_available():
                try:
                    if dd.kind == "imagem":
                        disp = im["disp"]
                        lines = vision.new_text_in_amplified(disp, cfg, scale=im["disp_scale"], normal=pd.ocr_lines)
                    else:
                        lines = vision.new_text_in_amplified(pil, cfg, scale=scale, origin=origin, normal=normal)
                except Exception:
                    lines = []
                if lines:
                    ev = ["texto de baixo contraste revelado por realce de imagem"]
                    n = _pattern_findings(ctx, "visual", lines, pd.number, "imagem", True, True, extra_evidence=ev)
                    if n == 0:
                        bb = (min(u.bbox[0] for u in lines), min(u.bbox[1] for u in lines),
                              max(u.bbox[2] for u in lines), max(u.bbox[3] for u in lines))
                        ctx.add("visual", pd.number, "imagem", "Texto de baixo contraste em imagem", "Conteúdo escondido", "media",
                                True, " ".join(u.text for u in lines), bb, "imagem embutida", ev)
            msg = vision.lsb_message(pil, cfg.lsb_min_chars)
            if msg:
                hit = match_units([Unit(text=msg)], ctx.cpats, True)
                ctx.add("visual", pd.number, "imagem", "Mensagem oculta em bits menos significativos (LSB)", "Esteganografia",
                        "critica" if hit else "alta", True, msg, b, "imagem embutida" if dd.kind == "pdf" else "imagem",
                        ["texto ASCII legível extraído dos bits menos significativos dos pixels"])


def agent_metadados(ctx: Ctx):
    dd = ctx.dd
    for u in dd.meta_units:
        n = _pattern_findings(ctx, "metadados", [u], u.page, "metadado", True if u.hidden or not u.bbox else u.hidden, True,
                              locator=u.locator)
        if n == 0 and u.reasons:
            ctx.add("metadados", u.page, "metadado", "Conteúdo escondido na estrutura do arquivo", "Conteúdo escondido", "media",
                    True, u.text, u.bbox, u.locator, list(u.reasons))
        _unicode_findings(ctx, "metadados", u, u.page, "metadado", u.hidden)
    for xref, snip in dd.extra.get("scripts", []):
        ctx.add("metadados", 1, "metadado", "Script ou ação executável embutida", "Conteúdo ativo", "media", True, snip, None,
                f"objeto PDF {xref}", ["JavaScript, Launch ou ação automática dentro do PDF"])
    for name, size in dd.extra.get("embedded", []):
        ctx.add("metadados", 1, "metadado", "Arquivo anexado ao PDF", "Conteúdo escondido", "baixa", True,
                f"{name} ({size} bytes)", None, f"anexo {name}", ["arquivos anexos não aparecem na leitura normal"])
    if dd.extra.get("ocg_off"):
        ctx.add("metadados", 1, "metadado", "Camadas (OCG) desativadas", "Conteúdo escondido", "baixa", True,
                ", ".join(dd.extra["ocg_off"]), None, "camadas opcionais", ["camadas desligadas por padrão no visualizador"])


def agent_semantico(ctx: Ctx):
    cfg = ctx.cfg
    hits = probe_document(ctx.store, ctx.embedder, ctx.doc_id, ctx.probes, cfg)
    for h in hits:
        meta = h["meta"]
        page = int(meta["page"])
        bbox = json.loads(meta["bbox"]) if meta.get("bbox") else None
        htoks = vision.norm_tokens(h["text"])
        matched = None
        for f in ctx.findings:
            if f.page != page:
                continue
            ftoks = vision.norm_tokens(f.snippet)
            if ftoks and len(ftoks & htoks) / len(ftoks) >= 0.5:
                matched = f
                break
        ev = [f"similaridade {h['sim']:.2f} com a sonda: {h['probe'][:70]}"]
        if matched:
            matched.corroborated += 1
            matched.evidence += ev
            continue
        hid = bool(meta.get("hidden"))
        ctx.add("semantico", page, meta["modality"], "Similaridade semântica com comando conhecido", "Similaridade semântica",
                "alta" if hid else "media", hid, h["text"], bbox, meta.get("locator", ""), ev)


AGENTS = {"texto": agent_texto, "visual": agent_visual, "metadados": agent_metadados, "semantico": agent_semantico}


def corroborate(ctx: Ctx):
    """Verificação cruzada: procura no banco vetorial o mesmo comando repetido em outros pontos do processo."""
    cfg = ctx.cfg
    n_new = 0
    for f in ctx.findings:
        if f.severity not in ("critica", "alta") or not f.snippet:
            continue
        q = ctx.embedder.embed([f.snippet])[0]
        ftoks = vision.norm_tokens(f.snippet)
        seen_pages = set()
        for h in ctx.store.query(q, cfg.top_k, doc_id=ctx.doc_id):
            if h["sim"] < cfg.corroborate_threshold:
                continue
            pg = int(h["meta"]["page"])
            htoks = vision.norm_tokens(h["text"])
            same_place = pg == f.page and ftoks and len(ftoks & htoks) / len(ftoks) >= 0.6
            if same_place or pg in seen_pages:
                continue
            seen_pages.add(pg)
            f.corroborated += 1
            f.evidence.append(f"trecho semelhante também na pág. {pg} (similaridade {h['sim']:.2f})")
            n_new += 1
    return n_new


SEV_RANK = {"critica": 4, "alta": 3, "media": 2, "baixa": 1}


def dedupe(findings: list[Finding]) -> list[Finding]:
    """Remove repetições e funde, num único achado, as regras que apontam para o mesmo ponto do documento."""
    seen, uniq = set(), []
    for f in findings:
        key = (f.page, f.rule, " ".join(f.snippet.lower().split())[:80])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(f)
    groups: dict = {}
    order = []
    for f in uniq:
        place = tuple(f.bbox) if f.bbox else f.locator
        key = (f.agent, f.page, place, f.hidden, f.modality)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(f)
    out = []
    for key in order:
        g = groups[key]
        main = max(g, key=lambda x: SEV_RANK.get(x.severity, 0))
        for o in g:
            if o is main:
                continue
            if o.rule != main.rule:
                main.evidence.append(f"também enquadrado em: {o.rule}")
            main.evidence += [e for e in o.evidence if e not in main.evidence]
            main.corroborated = max(main.corroborated, o.corroborated)
        out.append(main)
    return out


def run_agents(ctx: Ctx) -> list[Finding]:
    plan = [a for a in ("texto", "visual", "metadados", "semantico") if a in ctx.cfg.enabled_agents]
    ctx.log("orquestrador", "plano", " > ".join(plan) + (" > verificação cruzada" if ctx.cfg.corroborate else ""))
    for name in plan:
        t0 = time.time()
        before = len(ctx.findings)
        try:
            AGENTS[name](ctx)
            ctx.log(name, "executou", f"{len(ctx.findings) - before} achados novos", (time.time() - t0) * 1000)
        except Exception as e:  # um agente com falha não derruba os demais
            ctx.log(name, "falhou", repr(e)[:200], (time.time() - t0) * 1000)
    if ctx.cfg.corroborate:
        t0 = time.time()
        n = corroborate(ctx)
        ctx.log("orquestrador", "verificação cruzada", f"{n} corroborações no banco vetorial", (time.time() - t0) * 1000)
    ctx.findings = dedupe(ctx.findings)
    return ctx.findings
