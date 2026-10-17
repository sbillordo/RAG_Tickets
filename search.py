"""Dado un caso nuevo, busca tickets parecidos Y documentación relevante.

Uso:
    python search.py "no puedo imprimir desde el portátil, sale error de spooler"
    python search.py --ticket 15000          # usa un ticket existente de Zendesk como consulta
    python search.py --archivo caso.txt      # texto largo desde un fichero
    python search.py "texto" --tag sistemes --desde 2023 -k 8 --kdocs 3

La primera línea del texto se trata como el asunto y el resto como la descripción.

Tickets: búsqueda HÍBRIDA = embeddings (significado, solo con el asunto) + BM25 (palabras exactas, con
asunto y descripción), fusionadas con RRF. Medida con tickets fusionados: ver evaluate.py.
Documentación: solo embeddings (aún sin evaluar).

--tag filtra por tag (tickets) y por categoría (documentación).
--desde (año) solo se aplica a tickets: la documentación vigente vale aunque sea antigua.
--solo-finalizadas limita la documentación a las páginas '03. FINALIZADA' (por defecto salen todas,
con su estado visible: una página 'en desarrollo' puede ser justo la respuesta).
--modo semantico  usa solo embeddings en tickets (comportamiento anterior).
"""
import argparse
import os
import pickle
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import chromadb
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from index_chroma import COLLECTION, DB_DIR, MODEL_NAME, NOTION_COLLECTION
from ingestion.zendesk import get_users_zendesk
from processing.cleaning import clean_body
from processing.pseudonymizer import Pseudonymizer
from processing.sparse import rank_tickets, rrf
from redact_pass import load_exclude

load_dotenv()
ZENDESK_SUBDOMAIN = os.getenv("ZENDESK_SUBDOMAIN")
BM25_PATH = Path("data/bm25_zendesk.pkl")


# --- recursos pesados: se cargan una sola vez por proceso (la UI hace muchas búsquedas seguidas) ---
@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _client():
    return chromadb.PersistentClient(path=DB_DIR)


@lru_cache(maxsize=1)
def _bm25():
    return pickle.load(BM25_PATH.open("rb")) if BM25_PATH.exists() else None


@lru_cache(maxsize=1)
def _pseudo():
    users = get_users_zendesk()  # una llamada a Zendesk; reiniciar para recoger usuarios nuevos
    return Pseudonymizer(users=users, exclude_parts=load_exclude(users))


def warmup():
    """Carga todo por adelantado para que la primera búsqueda no tarde."""
    _pseudo(), _model(), _client(), _bm25()


def ticket_link(ticket_id) -> str:
    """Enlace que se abre en el navegador (el guardado en el índice es el de la API)."""
    return f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/agent/tickets/{ticket_id}"


def ticket_text(ticket_id: int) -> str:
    """Asunto + descripción de un ticket de Zendesk (texto sin tratar; se seudonimiza en search())."""
    from ingestion.zendesk import _get_client
    t = _get_client().tickets(id=ticket_id)
    return f"{t.subject}\n\n{t.description}"


def _where(conditions: list[dict]):
    return conditions[0] if len(conditions) == 1 else ({"$and": conditions} if conditions else None)


def _split(text: str) -> tuple[str, str]:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return (lines[0][:300], "\n".join(lines[1:])) if lines else ("", "")


def _dense_tickets(client, emb, where, n_chunks=120):
    """Tickets ordenados por significado, con su mejor similitud coseno."""
    res = client.get_collection(COLLECTION).query(query_embeddings=emb, n_results=n_chunks, where=where)
    order, cos = [], {}
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        t = int(meta["ticket_id"])
        if t not in cos:
            cos[t] = 1 - dist
            order.append(t)
    return order, cos


def _docs(client, emb, k, where):
    try:
        col = client.get_collection(NOTION_COLLECTION)
    except Exception:
        return []
    res = col.query(query_embeddings=emb, n_results=k * 6, where=where)
    best = {}  # varios fragmentos del mismo documento: se queda el mejor
    for meta, dist, doc in zip(res["metadatas"][0], res["distances"][0], res["documents"][0]):
        if meta["doc_id"] not in best:
            best[meta["doc_id"]] = (1 - dist, meta, doc)
    return sorted(best.values(), key=lambda x: -x[0])[:k]


def search(text: str, k: int = 5, kdocs: int = 3, tag: str | None = None, desde: int | None = None,
           solo_finalizadas: bool = False, excluir_ticket: int | None = None, modo: str = "hibrido"):
    # El caso nuevo se seudonimiza igual que el corpus (mismas exclusiones): sus marcadores coinciden con los del índice.
    pseudo = _pseudo()
    subject, desc = _split(text)
    subject_q = clean_body(pseudo.text(subject))
    desc_q = clean_body(pseudo.text(desc))
    full_q = f"{subject_q}\n\n{desc_q}".strip()

    model, client = _model(), _client()

    t_where = ([{f"tag__{tag}": True}] if tag else []) + ([{"year": {"$gte": desde}}] if desde else [])
    if excluir_ticket:
        t_where.append({"ticket_id": {"$ne": excluir_ticket}})
    d_where = ([{f"tag__{tag}": True}] if tag else []) + ([{"estado": "03. FINALIZADA"}] if solo_finalizadas else [])

    # --- tickets ---
    use_bm25 = modo == "hibrido" and _bm25() is not None
    if modo == "hibrido" and not use_bm25:
        print("(aviso: falta data/bm25_zendesk.pkl, ejecuta build_bm25.py; uso solo embeddings)")
    # Embeddings: híbrido -> solo asunto (lo que mejor funcionó); semántico -> asunto + descripción (como antes)
    dense_q = subject_q if use_bm25 else full_q
    emb_t = model.encode(["query: " + (dense_q or full_q)], normalize_embeddings=True).tolist()
    dense_order, cos = _dense_tickets(client, emb_t, _where(t_where))

    sparse_order, sparse_rank, data = [], {}, None
    if use_bm25:
        data = _bm25()
        mask = np.ones(len(data["ticket_id"]), dtype=bool)
        if excluir_ticket:
            mask &= data["ticket_id"] != excluir_ticket
        if tag:
            mask &= np.array([tag in s for s in data["tags"]])
        if desde:
            mask &= data["year"] >= desde
        sparse_order = rank_tickets(data["bm25"].scores(full_q), data["ticket_id"], mask)
        sparse_rank = {t: i + 1 for i, t in enumerate(sparse_order)}
    dense_rank = {t: i + 1 for i, t in enumerate(dense_order)}
    fused = rrf([dense_order, sparse_order]) if use_bm25 else dense_order

    tickets = []
    for t in fused[:k]:
        if data:
            m = data["tickets"][t]
        else:  # modo semántico: los metadatos están en Chroma
            m = client.get_collection(COLLECTION).get(where={"ticket_id": t}, limit=1, include=["metadatas"])["metadatas"][0]
            m = {"title": m["title"], "status": m["status"], "tags": m["tags"].split(", "), "created_ts": m["created_ts"]}
        tickets.append({"id": t, **m, "rank_sem": dense_rank.get(t), "cos": cos.get(t), "rank_pal": sparse_rank.get(t)})

    # --- documentación (solo embeddings) ---
    emb_d = model.encode(["query: " + full_q], normalize_embeddings=True).tolist()
    return {"tickets": tickets, "docs": _docs(client, emb_d, kdocs, _where(d_where)), "hibrido": use_bm25}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("texto", nargs="?", help="texto del caso (o usa --ticket / --archivo)")
    ap.add_argument("--ticket", type=int, help="ID de un ticket de Zendesk para usar como consulta")
    ap.add_argument("--archivo", help="fichero de texto con el caso")
    ap.add_argument("-k", type=int, default=5, help="nº de tickets")
    ap.add_argument("--kdocs", type=int, default=3, help="nº de documentos")
    ap.add_argument("--tag")
    ap.add_argument("--desde", type=int, help="año mínimo para tickets, p. ej. 2023")
    ap.add_argument("--solo-finalizadas", action="store_true")
    ap.add_argument("--modo", choices=["hibrido", "semantico"], default="hibrido")
    a = ap.parse_args()

    if a.ticket:
        texto = ticket_text(a.ticket)
    elif a.archivo:
        texto = open(a.archivo, encoding="utf-8").read()
    elif a.texto:
        texto = a.texto
    else:
        ap.error("indica el texto del caso, --ticket ID o --archivo FICHERO")
    out = search(texto, a.k, a.kdocs, a.tag, a.desde, a.solo_finalizadas, excluir_ticket=a.ticket, modo=a.modo)

    print(f"\n=== CASOS PARECIDOS (Zendesk, {'búsqueda híbrida' if out['hibrido'] else 'solo embeddings'}) ===")
    for n, t in enumerate(out["tickets"], start=1):
        fecha = datetime.fromtimestamp(t["created_ts"]).strftime("%Y-%m-%d")
        por = []
        if t["rank_sem"]:
            por.append(f"significado #{t['rank_sem']} ({t['cos']:.2f})")
        if t["rank_pal"]:
            por.append(f"palabras #{t['rank_pal']}")
        print(f"\n{n}. #{t['id']}  {fecha}  {t['status']}  {t['title']}")
        print(f"   tags: {', '.join(t['tags'])}   [{' · '.join(por)}]\n   {ticket_link(t['id'])}")

    print("\n\n=== DOCUMENTACIÓN RELEVANTE (Notion) ===")
    if not out["docs"]:
        print("(sin resultados: ¿está indexada la colección notion_docs?)")
    for n, (score, m, doc) in enumerate(out["docs"], start=1):
        seccion = next((l[len("Sección: "):] for l in doc.split("\n") if l.startswith("Sección: ")), "")
        print(f"\n{n}. [{score:.3f}] {m.get('doc_ref', '')}  {m['title']}   ({m.get('estado') or 'sin estado'})")
        print(f"   categorías: {m['tags']}" + (f"\n   sección: {seccion}" if seccion else ""))
        print(f"   {m['url']}")
