"""Camada 3: banco vetorial. ChromaDB persistente (coleção de teste) com fallback em NumPy."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from .config import DATA_DIR


class ChromaStore:
    kind = "chromadb"

    def __init__(self, name: str):
        import chromadb
        from chromadb.config import Settings
        self.client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma"), settings=Settings(anonymized_telemetry=False))
        self.name = name
        self.col = self.client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def add(self, ids, embeddings, documents, metadatas):
        for i in range(0, len(ids), 200):
            self.col.add(ids=ids[i:i + 200], embeddings=embeddings[i:i + 200].tolist(),
                         documents=documents[i:i + 200], metadatas=metadatas[i:i + 200])

    def count(self, doc_id: str | None = None) -> int:
        if doc_id is None:
            return self.col.count()
        return len(self.col.get(where={"doc_id": doc_id}, include=[])["ids"])

    def query(self, embedding, top_k: int, doc_id: str | None = None):
        where = {"doc_id": doc_id} if doc_id else None
        n = self.count(doc_id)
        if n == 0:
            return []
        res = self.col.query(query_embeddings=[list(map(float, embedding))], n_results=min(top_k, n), where=where,
                             include=["documents", "metadatas", "distances"])
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({"text": doc, "meta": meta, "sim": 1.0 - float(dist)})
        return out

    def delete_doc(self, doc_id: str):
        self.col.delete(where={"doc_id": doc_id})

    def reset(self):
        self.client.delete_collection(self.name)
        self.col = self.client.get_or_create_collection(name=self.name, metadata={"hnsw:space": "cosine"})

    def list_collections(self) -> list[dict]:
        return [{"nome": c.name, "itens": c.count()} for c in self.client.list_collections()]


class NumpyStore:
    kind = "numpy"

    def __init__(self, name: str):
        self.path = DATA_DIR / f"{name}.pkl"
        self.name = name
        self.ids, self.docs, self.metas = [], [], []
        self.emb = np.zeros((0, 1), dtype=np.float32)
        if self.path.exists():
            self.ids, self.docs, self.metas, self.emb = pickle.loads(self.path.read_bytes())

    def _save(self):
        self.path.write_bytes(pickle.dumps((self.ids, self.docs, self.metas, self.emb)))

    def add(self, ids, embeddings, documents, metadatas):
        self.emb = embeddings if len(self.ids) == 0 else np.vstack([self.emb, embeddings])
        self.ids += list(ids)
        self.docs += list(documents)
        self.metas += list(metadatas)
        self._save()

    def count(self, doc_id: str | None = None) -> int:
        return len(self.ids) if doc_id is None else sum(m["doc_id"] == doc_id for m in self.metas)

    def query(self, embedding, top_k: int, doc_id: str | None = None):
        if not self.ids:
            return []
        idx = [i for i, m in enumerate(self.metas) if doc_id is None or m["doc_id"] == doc_id]
        if not idx:
            return []
        sims = self.emb[idx] @ np.asarray(embedding, dtype=np.float32)
        order = np.argsort(-sims)[:top_k]
        return [{"text": self.docs[idx[o]], "meta": self.metas[idx[o]], "sim": float(sims[o])} for o in order]

    def delete_doc(self, doc_id: str):
        keep = [i for i, m in enumerate(self.metas) if m["doc_id"] != doc_id]
        self.ids = [self.ids[i] for i in keep]
        self.docs = [self.docs[i] for i in keep]
        self.metas = [self.metas[i] for i in keep]
        self.emb = self.emb[keep] if keep else np.zeros((0, 1), dtype=np.float32)
        self._save()

    def reset(self):
        self.ids, self.docs, self.metas = [], [], []
        self.emb = np.zeros((0, 1), dtype=np.float32)
        self._save()

    def list_collections(self) -> list[dict]:
        return [{"nome": self.name, "itens": len(self.ids)}]


def collection_name(cfg, embedder) -> str:
    return f"{cfg.collection}_{embedder.name}{embedder.dim}"


def open_store(cfg, embedder):
    name = collection_name(cfg, embedder)
    try:
        return ChromaStore(name)
    except Exception:
        return NumpyStore(name)
