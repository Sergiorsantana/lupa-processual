"""Estruturas de dados compartilhadas entre as camadas."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict

BBox = tuple  # (x0, y0, x1, y1) em unidades da página


@dataclass
class Run:
    """Trecho de texto com atributos de renderização (camada 1)."""
    text: str
    bbox: BBox
    size: float = 0.0
    color: tuple = (0.0, 0.0, 0.0)
    opacity: float = 1.0
    mode: int = 0
    layer: str | None = None
    seqno: int = 0
    reasons: list = field(default_factory=list)


@dataclass
class Unit:
    """Unidade localizável de texto: bloco visível, grupo oculto, linha de OCR, parágrafo."""
    text: str
    bbox: BBox | None = None
    locator: str = ""
    hidden: bool = False
    reasons: list = field(default_factory=list)
    modality: str = "texto"  # texto | imagem | metadado
    page: int = 1


@dataclass
class PageData:
    number: int
    width: float
    height: float
    image_path: str | None = None
    scale: float = 1.0                       # pixels da imagem por unidade da página
    runs: list = field(default_factory=list)       # list[Run]
    blocks: list = field(default_factory=list)     # list[Unit] texto visível
    ocr_lines: list = field(default_factory=list)  # list[Unit] linhas do OCR da página renderizada
    images: list = field(default_factory=list)     # list[dict] imagens embutidas: {"xref","bbox","pil"}
    text_layer: str = ""
    ocr_all: str = ""                              # todo o texto que o OCR leu na página renderizada


@dataclass
class DocData:
    name: str
    kind: str  # pdf | docx | txt | imagem
    path: str
    pages: list = field(default_factory=list)  # list[PageData]
    meta_units: list = field(default_factory=list)  # list[Unit] metadados, anotações, anexos
    paragraph_units: list = field(default_factory=list)  # docx/txt: Unit com locator
    extra: dict = field(default_factory=dict)


@dataclass
class Finding:
    doc_id: str
    doc_name: str
    page: int
    modality: str           # texto | imagem | metadado
    agent: str
    rule: str
    category: str
    severity: str           # critica | alta | media | baixa
    hidden: bool
    snippet: str
    bbox: list | None = None
    page_w: float = 0.0
    page_h: float = 0.0
    locator: str = ""
    evidence: list = field(default_factory=list)
    corroborated: int = 0
    confidence: float = 0.0
    status: str = "ativo"   # ativo | descartado
    num: int = 0
    match: str = ""          # trecho exato que casou com a regra (usado para localizar)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Finding":
        return Finding(**d)
