"""Testes de ponta a ponta com os documentos de exemplo. Rode com: pytest -q"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("HCD_DATA", tempfile.mkdtemp(prefix="hcd_test_"))

from core import samples  # noqa: E402
from core.config import load_config, load_patterns, load_probes  # noqa: E402
from core.embeddings import get_embedder  # noqa: E402
from core.pipeline import process_file  # noqa: E402
from core.report import build_report  # noqa: E402
from core.storage import DB  # noqa: E402
from core.vectorstore import open_store  # noqa: E402

CFG = load_config()
EMB = get_embedder(CFG)
STORE = open_store(CFG, EMB)
DB_ = DB()
OUT = Path(os.environ["HCD_DATA"]) / "samples"


def run(path):
    return process_file(str(path), CFG, load_patterns(), load_probes(), STORE, EMB, DB_)


def test_pdf_com_comandos():
    OUT.mkdir(parents=True, exist_ok=True)
    r = run(samples.make_pdf_com_comandos(OUT / "a.pdf"))
    assert r["doc"]["verdict"] == "NÃO CONFORME"
    motivos = " ".join(e for f in r["findings"] for e in f.evidence)
    for m in ("cor igual ao fundo", "fonte minúscula", "fora da área", "coberto por forma", "Tr 3", "anotação com flag"):
        assert m in motivos, m
    assert any("LSB" in f.rule for f in r["findings"])
    assert any("baixo contraste" in " ".join(f.evidence) for f in r["findings"])
    assert all(f.bbox for f in r["findings"] if f.modality == "texto")


def test_pdf_limpo_sem_falso_positivo():
    OUT.mkdir(parents=True, exist_ok=True)
    r = run(samples.make_pdf_limpo(OUT / "b.pdf"))
    assert r["doc"]["verdict"] == "CONFORME"
    assert not r["findings"]


def test_docx_e_imagem():
    OUT.mkdir(parents=True, exist_ok=True)
    r = run(samples.make_docx(OUT / "c.docx"))
    assert r["doc"]["verdict"] == "NÃO CONFORME"
    assert any("tag Unicode" in f.rule for f in r["findings"])
    r = run(samples.make_imagem(OUT / "d.png"))
    assert r["doc"]["verdict"] == "NÃO CONFORME"


def test_relatorio_pdf():
    docs = DB_.documents()
    pdf = build_report(docs, {d["id"]: DB_.findings(d["id"]) for d in docs}, CFG.key_params())
    assert pdf[:4] == b"%PDF" and len(pdf) > 10_000
