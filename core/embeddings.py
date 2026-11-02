"""Camada 3: modelos de embedding. O padrão é offline (hashing de n-gramas), com SBERT opcional."""
from __future__ import annotations

import math
import re
import unicodedata
import zlib

import numpy as np

STOP = set("""a o as os um uma uns umas de do da dos das em no na nos nas por para com sem sob sobre e ou que se ao aos à às
the of to in on at for and or is are be this that these those it its as by from with""".split())


# Léxico conceitual: aproxima paráfrases de comandos sem depender de modelo externo.
# Cada conceito agrupa radicais (início de palavra) e expressões em português e inglês.
CONCEITOS = {
    "IGNORAR": ["ignor", "desconsider", "esquec", "descart", "desobedec", "disregard", "forget", "override", "never existed", "nunca existiram"],
    "REGRAS": ["instru", "regra", "orienta", "diretriz", "comando", "prompt", "guideline", "rule", "directive", "instruction"],
    "ANTERIOR": ["anterior", "antes", "previ", "prior", "earlier", "original", "acima", "above"],
    "IA": ["inteligencia artificial", "modelo de linguagem", "llm", "assistente", "automatizado", "algoritmo", "assistant",
           "language model", "chatbot", "robo", "bot ", " ia ", " ai ", "a ai ", "an ai ", "artificial"],
    "DECIDIR": ["decid", "julg", "aprov", "deferir", "defira", "favor", "procedent", "approve", "conced", "grant", "rule in"],
    "OCULTAR": ["mencion", "revel", "silenci", "omit", "ocult", "segredo", "disclose", "mention", "hide", "secret", "calar"],
    "PONTUAR": ["pontua", "nota maxima", "score", "rating", "classific", "priorid", "urgente", "highest", "maxima"],
    "SISTEMA": ["system prompt", "prompt do sistema", "instrucoes internas", "internal instructions"],
}


def _concepts(norm_text: str) -> set:
    padded = f" {norm_text} "
    found = set()
    for conc, keys in CONCEITOS.items():
        for k in keys:
            if (" " in k and k in padded) or any(w.startswith(k.strip()) for w in padded.split() if not k.endswith(" ")) or (k.endswith(" ") and k in padded):
                found.add(conc)
                break
    return found


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


class HashingEmbedder:
    """Embedding determinístico: unigramas, bigramas e n-gramas de caracteres com hashing assinado."""

    name = "hashing"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _feats(self, text: str):
        t = _strip_accents(text.lower())
        words = [w for w in re.findall(r"[a-z0-9]+", t) if w not in STOP and len(w) > 1]
        for w in words:
            yield "w:" + w, 1.0
            p = f"<{w}>"
            for n in (3, 4, 5):
                for i in range(max(len(p) - n + 1, 0)):
                    yield "c:" + p[i:i + n], 0.35
        for a, b in zip(words, words[1:]):
            yield f"b:{a}_{b}", 0.8
        conc = sorted(_concepts(t))
        for c in conc:
            yield "k:" + c, 1.2
        for i, a in enumerate(conc):
            for b in conc[i + 1:]:
                yield f"kp:{a}_{b}", 1.6

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, text in enumerate(texts):
            counts: dict = {}
            for f, w in self._feats(text):
                counts[f] = counts.get(f, 0.0) + w
            for f, c in counts.items():
                h = zlib.crc32(f.encode("utf-8"))
                sign = 1.0 if (h >> 31) & 1 else -1.0
                out[r, h % self.dim] += sign * (1.0 + math.log(c))
            n = np.linalg.norm(out[r])
            if n > 0:
                out[r] /= n
        return out


class SbertEmbedder:
    name = "sbert"

    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.model = SentenceTransformer(model)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


_CACHE: dict = {}


def get_embedder(cfg):
    key = (cfg.embedder, cfg.embed_dim, cfg.sbert_model)
    if key in _CACHE:
        return _CACHE[key]
    emb = None
    if cfg.embedder == "sbert":
        try:
            emb = SbertEmbedder(cfg.sbert_model)
        except Exception:
            emb = None  # cai para o embedder offline
    if emb is None:
        emb = HashingEmbedder(cfg.embed_dim)
    _CACHE[key] = emb
    return emb
