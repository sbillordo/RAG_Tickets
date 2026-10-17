"""Fase 2b: data/chunks.jsonl -> embeddings locales -> Chroma (chroma_db/)

Uso:  python index_chroma.py [--source zendesk|notion] [--limit N] [--rebuild]

Es reanudable: por defecto salta los fragmentos que ya están en Chroma, así que si se
interrumpe basta con volver a lanzarlo. --rebuild borra la colección y empieza de cero.
"""
import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"  # multilingüe (es/ca/en), 384 dimensiones
DB_DIR = "chroma_db"
COLLECTION = "zendesk_tickets"        # tickets (se mantiene el nombre: lo importa search.py)
NOTION_COLLECTION = "notion_docs"
SOURCES = {  # fuente -> (fichero de fragmentos, colección)
    "zendesk": (Path("data/chunks.jsonl"), COLLECTION),
    "notion": (Path("data/notion_chunks.jsonl"), NOTION_COLLECTION),
}
BATCH = 64


def chroma_metadata(c: dict) -> dict:
    """Chroma solo admite str/int/float/bool: las tags/categorías van como booleanos tag__<nombre>."""
    m = c["metadata"]
    created = datetime.fromisoformat(c["created_at"])
    meta = {
        "doc_id": c["doc_id"],
        "title": c["title"][:300],
        "status": c["status"],
        "created_ts": int(created.timestamp()),
        "year": created.year,
        "tags": ", ".join(m.get("tags") or []),
    }
    if c.get("updated_at"):
        meta["updated_ts"] = int(datetime.fromisoformat(c["updated_at"]).timestamp())
    for key, value in m.items():  # el resto de metadatos escalares de la fuente, tal cual
        if key != "tags" and isinstance(value, (str, int, float, bool)) and value not in ("", None):
            meta.setdefault(key, value)
    if "ticket_id" in meta:
        meta["ticket_id"] = int(meta["ticket_id"])
    for tag in m.get("tags") or []:
        meta[f"tag__{tag}"] = True
    return meta


def main(source: str, limit: int | None, rebuild: bool, sync: bool = False,
         model_name: str = MODEL_NAME, collection: str | None = None):
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler("logs/index_chroma.log", encoding="utf-8"), logging.StreamHandler()],
    )
    log = logging.getLogger("index")
    t0 = time.monotonic()

    chunks_path, collection_name = SOURCES[source]
    collection_name = collection or collection_name   # colección alternativa para comparar modelos
    chunks =[json.loads(l) for l in chunks_path.open(encoding="utf-8")]
    if limit:
        chunks = chunks[:limit]

    client = chromadb.PersistentClient(path=DB_DIR)
    if rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    col = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    if sync:  # compara con lo indexado: borra fragmentos obsoletos/cambiados y deja pendientes los nuevos
        stored = col.get(include=["documents"])
        stored_text = dict(zip(stored["ids"], stored["documents"]))
        new_text = {c["chunk_id"]: c["text"] for c in chunks}
        stale = [i for i, t in stored_text.items() if new_text.get(i) != t]
        for j in range(0, len(stale), 2000):
            col.delete(ids=stale[j:j + 2000])
        log.info("SYNC  %d fragmentos obsoletos o modificados eliminados", len(stale))
    already = set(col.get(include=[])["ids"])
    total = len(chunks)
    chunks = [c for c in chunks if c["chunk_id"] not in already]
    # por longitud: lotes homogéneos = menos relleno = más rápido
    chunks.sort(key=lambda c: len(c["text"]))
    log.info("INICIO [%s -> %s]  %d fragmentos pendientes de %d (ya indexados: %d), modelo %s",
             source, collection_name, len(chunks), total, len(already), model_name)

    model = SentenceTransformer(model_name)
    model.max_seq_length = 512

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        emb = model.encode(
            ["passage: " + c["text"] for c in batch], normalize_embeddings=True, show_progress_bar=False
        )
        col.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=emb.tolist(),
            documents=[c["text"] for c in batch],
            metadatas=[chroma_metadata(c) for c in batch],
        )
        done = i + len(batch)
        if (i // BATCH) % 10 == 0:
            rate = done / (time.monotonic() - t0)
            log.info("%d/%d fragmentos (%.1f/s, quedan ~%d min)", done, len(chunks), rate,
                     (len(chunks) - done) / max(rate, 0.1) / 60)

    secs = int(time.monotonic() - t0)
    log.info("FIN  %d fragmentos indexados  duración=%dh %02dm %02ds", col.count(), secs // 3600,
             secs % 3600 // 60, secs % 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=SOURCES, default="zendesk")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rebuild", action="store_true", help="borra la colección y empieza de cero")
    ap.add_argument("--sync", action="store_true", help="solo reindexa los fragmentos que han cambiado")
    ap.add_argument("--model", default=MODEL_NAME, help="modelo de embeddings (para comparar modelos)")
    ap.add_argument("--collection", default=None, help="nombre de colección alternativo (para comparar modelos)")
    a = ap.parse_args()
    main(a.source, a.limit, a.rebuild, a.sync, a.model, a.collection)
