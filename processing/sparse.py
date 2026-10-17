"""Búsqueda por palabras (BM25) para combinarla con los embeddings (búsqueda híbrida).

Los embeddings captan el significado ("monitor negro" ~ "pantalla no enciende"); BM25 capta las palabras
exactas y raras (nombres de sistemas, códigos de error, "vCenter", "BLE"). Se fusionan con RRF.
"""
import pickle
import re
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

_MARKERS = re.compile(r"\[(?:USUARIO|EMAIL|TEL|DOC|IBAN|SECRETO)(?:_\w+)?\]")


def _prep(text: str) -> str:
    return _MARKERS.sub(" ", text).lower()


class Bm25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, texts: list[str]):
        self.vec = CountVectorizer(
            preprocessor=_prep, strip_accents="unicode", token_pattern=r"(?u)\b[a-z0-9][a-z0-9_.-]{2,}\b",
            dtype=np.float32,
        )
        tf = self.vec.fit_transform(texts).tocoo()
        n = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avg = dl.mean()
        df = np.bincount(tf.col, minlength=tf.shape[1])
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        data = tf.data * (self.k1 + 1) / (tf.data + self.k1 * (1 - self.b + self.b * dl[tf.row] / avg)) * idf[tf.col]
        self.W = csr_matrix((data, (tf.row, tf.col)), shape=tf.shape)
        return self

    def scores(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])
        q.data[:] = 1.0
        return np.asarray((self.W @ q.T).todense()).ravel()

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "Bm25":
        with open(path, "rb") as f:
            return pickle.load(f)


def rank_tickets(scores: np.ndarray, ticket_ids: np.ndarray, mask: np.ndarray, top: int = 120) -> list[int]:
    """Orden de tickets (mejor fragmento de cada uno) entre los fragmentos permitidos por `mask`."""
    idx = np.where(mask & (scores > 0))[0]
    idx = idx[np.argsort(-scores[idx])]
    seen: list[int] = []
    for i in idx:
        t = int(ticket_ids[i])
        if t not in seen:
            seen.append(t)
            if len(seen) >= top:
                break
    return seen


def rrf(rankings: list[list[int]], weights: list[float] | None = None, k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: combina varias listas ordenadas sin mezclar escalas de score."""
    weights = weights or [1.0] * len(rankings)
    score: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for pos, t in enumerate(ranking, start=1):
            score[t] = score.get(t, 0.0) + w / (k + pos)
    return sorted(score, key=score.get, reverse=True)
