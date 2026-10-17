"""Evaluación del buscador con tickets fusionados (closed_by_merge).

Idea: si un agente fusionó el ticket A en el B, eran el MISMO problema. Se usa A como consulta
y se mide en qué posición aparece B entre los resultados (excluyendo al propio A).

Escenario REALISTA: solo pares con destino anterior al origen, solo candidatos anteriores en el tiempo,
y se descartan los pares donde el destino aún contiene una copia del origen (fuga).

Uso:  python evaluate.py [--modes base,sin_firma,...] [--collection zendesk_tickets] [--tag-ruido]
Guarda data/eval_merge.csv (solo IDs y posiciones, sin texto).
"""
import argparse
import csv
import json
import re
import statistics as st
from datetime import datetime

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from index_chroma import COLLECTION, DB_DIR, MODEL_NAME
from processing.cleaning import clean_body, parse_content, strip_markers, strip_signature
from processing.sparse import Bm25, rank_tickets, rrf

MERGE_RE = re.compile(
    r"(?i)(?:fusionad[ao]\s+en\s+la\s+solicitud|merged\s+into\s+request|combinad[ao]\s+(?:con|en)\s+(?:la\s+)?solicitud"
    r"|fusionat\w*\s+(?:a|amb)\s+la\s+sol[·.]?licitud)\s*#\s?(\d+)"
)
KS = (1, 5, 10, 20)


# ---- estrategias de consulta: cada una recibe (asunto, descripción limpia) ----
def q_base(subj, desc):
    return f"{subj}\n\n{desc}"


def q_asunto(subj, desc):
    return subj


def q_sin_firma(subj, desc):
    return f"{subj}\n\n{strip_signature(desc)}"


def q_sin_marcadores(subj, desc):
    return strip_markers(f"{subj}\n\n{strip_signature(desc)}")


def q_corta(subj, desc):
    return strip_markers(f"{subj}\n\n{strip_signature(desc)[:600]}")


def q_muy_corta(subj, desc):
    return strip_markers(f"{subj}\n\n{strip_signature(desc)[:300]}")


MODES = {
    # nombre: (consulta para embeddings, consulta para BM25, pesos [denso, bm25])
    "base": ("base", None, None),                    # lo que hace search.py hoy
    "asunto": ("asunto", None, None),
    "bm25_base": (None, "base", None),
    "bm25_asunto": (None, "asunto", None),
    "hibrido_base": ("base", "base", [1.0, 1.0]),
    "hibrido_asunto": ("asunto", "asunto", [1.0, 1.0]),
    "hibrido_asunto+base": ("asunto", "base", [1.0, 1.0]),
    "hibrido_asunto+base_denso2": ("asunto", "base", [2.0, 1.0]),
}
QUERY_FNS = {"base": q_base, "asunto": q_asunto, "sin_firma": q_sin_firma, "corta_600": q_corta}


def _words(subject: str) -> set:
    subject = re.sub(r"(?i)\b(re|rv|fw|fwd)\s*:", " ", subject)
    return set(re.findall(r"[a-z0-9]{3,}", subject.lower()))


def subjects_differ(a: str, b: str, thr: float = 0.5) -> bool:
    """True si los asuntos comparten menos de la mitad de sus palabras (no es el mismo hilo de correo)."""
    wa, wb = _words(a), _words(b)
    return not wa or not wb or len(wa & wb) / len(wa | wb) < thr


def load_pairs(docs):
    by_id = {d["metadata"]["ticket_id"]: d for d in docs}
    pairs = []
    for d in docs:
        if "closed_by_merge" not in d["metadata"]["tags"]:
            continue
        targets = set(MERGE_RE.findall(d["content"]))
        if len(targets) != 1:
            continue
        tid = int(targets.pop())
        if tid in by_id and tid != d["metadata"]["ticket_id"]:
            pairs.append((d, by_id[tid]))
    return pairs


def dense_ranking(col, emb, src_id, where, n_chunks=120):
    res = col.query(query_embeddings=[emb], n_results=n_chunks, where=where)
    seen = []
    for meta in res["metadatas"][0]:
        t = meta["ticket_id"]
        if t != src_id and t not in seen:
            seen.append(t)
    return seen


def position(ranking, target_id):
    return ranking.index(target_id) + 1 if target_id in ranking else None


def metrics(ranks):
    n = len(ranks)
    if not n:
        return {k: 0 for k in ("n", *[f"@{k}" for k in KS], "mrr", "miss")}
    out = {"n": n, "mrr": sum(1 / r for r in ranks if r) / n, "miss": sum(r is None for r in ranks) / n}
    for k in KS:
        out[f"@{k}"] = sum(r is not None and r <= k for r in ranks) / n
    return out


def line(name, ranks):
    m = metrics(ranks)
    print(f"{name:<34} n={m['n']:<4} @1 {100*m['@1']:3.0f}%  @5 {100*m['@5']:3.0f}%  @10 {100*m['@10']:3.0f}%  "
          f"@20 {100*m['@20']:3.0f}%  MRR {m['mrr']:.3f}  fuera del top120 {100*m['miss']:3.0f}%")


def main(modes, collection, model_name=MODEL_NAME, rerank_model=None, rerank_top=20):
    docs = [json.loads(l) for l in open("data/zendesk_docs.jsonl", encoding="utf-8")]
    pairs = []
    for s, t in load_pairs(docs):
        if int(datetime.fromisoformat(t["created_at"]).timestamp()) < int(datetime.fromisoformat(s["created_at"]).timestamp()):
            pairs.append((s, t))
    print(f"pares con destino anterior al origen: {len(pairs)}")

    print(f"modelo: {model_name} | colección: {collection}")
    model = SentenceTransformer(model_name)
    col = chromadb.PersistentClient(path=DB_DIR).get_collection(collection)

    # descarta pares donde el destino contiene una copia del origen
    clean_pairs = []
    for s, t in pairs:
        sid = s["metadata"]["ticket_id"]
        got = col.get(where={"ticket_id": t["metadata"]["ticket_id"]}, include=["documents"])
        if not re.search(rf"#\s?{sid}\b", "\n".join(got["documents"])):
            clean_pairs.append((s, t))
    print(f"pares sin copia del origen en el destino (usados): {len(clean_pairs)}\n")

    # índice de palabras (BM25) sobre los mismos fragmentos que Chroma
    chunks = [json.loads(l) for l in open("data/chunks.jsonl", encoding="utf-8")]
    bm25 = Bm25().fit([c["text"] for c in chunks])
    c_tid = np.array([int(c["metadata"]["ticket_id"]) for c in chunks])
    c_ts = np.array([int(datetime.fromisoformat(c["created_at"]).timestamp()) for c in chunks])

    parts = [(parse_content(s["content"]), s) for s, _ in clean_pairs]
    dense_cache = {}

    def texts_for(fn_name):
        fn = QUERY_FNS[fn_name]
        return [fn(p.subject, clean_body(p.description)) for p, _ in parts]

    # reranker (cross-encoder): relee consulta + candidato juntos y reordena los primeros `rerank_top`
    reranker = rr_query = ticket_doc = None
    if rerank_model:
        from sentence_transformers import CrossEncoder
        print(f"reranker: {rerank_model} (sobre los {rerank_top} primeros)")
        reranker = CrossEncoder(rerank_model, max_length=512)
        rr_query = [q[:1000] for q in texts_for("base")]
        ticket_doc = {}  # texto de cada ticket para el reranker: su primer fragmento (asunto + tags + descripción)
        for c in chunks:
            ticket_doc.setdefault(int(c["metadata"]["ticket_id"]), c["text"][:1500])

    rows, results = [], {}
    for mode in modes:
        extra = {}
        dq, sq, weights = MODES[mode]
        embs = None
        if dq:
            if dq not in dense_cache:
                dense_cache[dq] = model.encode(["query: " + q for q in texts_for(dq)], normalize_embeddings=True,
                                               batch_size=32).tolist()
            embs = dense_cache[dq]
        sparse_texts = texts_for(sq) if sq else None
        res = []
        for i, (s, t) in enumerate(clean_pairs):
            sid, tid = s["metadata"]["ticket_id"], t["metadata"]["ticket_id"]
            sts = int(datetime.fromisoformat(s["created_at"]).timestamp())
            lists = []
            if embs is not None:
                lists.append(dense_ranking(col, embs[i], sid, {"created_ts": {"$lt": sts}}))
            if sparse_texts is not None:
                sc = bm25.scores(sparse_texts[i])
                lists.append(rank_tickets(sc, c_tid, (c_ts < sts) & (c_tid != sid)))
            ranking = lists[0] if len(lists) == 1 else rrf(lists, weights)
            r = position(ranking, tid)
            same = s["metadata"]["requester"] == t["metadata"]["requester"]
            long_ = len(clean_body(parts[i][0].description)) > 1000
            diff = subjects_differ(parts[i][0].subject, parse_content(t["content"]).subject)
            res.append((r, same, long_, diff))
            rows.append((mode, sid, tid, r, int(same)))
            if reranker is not None:
                top, rest = ranking[:rerank_top], ranking[rerank_top:]
                scores = reranker.predict([(rr_query[i], ticket_doc.get(x, "")) for x in top],
                                          batch_size=32, show_progress_bar=False) if top else []
                reranked = [x for _, x in sorted(zip(scores, top), key=lambda z: -z[0])]
                for name, order in ((f"{mode}+rerank", reranked + rest),
                                    (f"{mode}+rerank_rrf", rrf([reranked, top]) + rest)):
                    rr = position(order, tid)
                    extra.setdefault(name, []).append((rr, same, long_, diff))
                    rows.append((name, sid, tid, rr, int(same)))
        results[mode] = res
        results.update(extra)

    with open("data/eval_merge.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["estrategia", "ticket_origen", "ticket_destino", "posicion", "mismo_solicitante"])
        w.writerows(rows)

    print("=== TODOS ===")
    for mode in results:
        line(mode, [x[0] for x in results[mode]])
    print("\n=== DISTINTO SOLICITANTE (el caso difícil y más representativo) ===")
    for mode in results:
        line(mode, [x[0] for x in results[mode] if not x[1]])
    print("\n=== DESCRIPCIÓN LARGA (>1000 caracteres) ===")
    for mode in results:
        line(mode, [x[0] for x in results[mode] if x[2]])


    print("\n=== ASUNTOS DISTINTOS entre origen y destino (descarta el mismo hilo de correo; la prueba MÁS exigente) ===")
    for mode in results:
        line(mode, [x[0] for x in results[mode] if x[3]])
    print("\n=== ASUNTOS PARECIDOS (mismo hilo; la prueba fácil) ===")
    for mode in results:
        line(mode, [x[0] for x in results[mode] if not x[3]])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--model", default=MODEL_NAME, help="debe ser el modelo con el que se indexó la colección")
    ap.add_argument("--rerank", default=None, help="modelo cross-encoder para reordenar (p. ej. cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)")
    ap.add_argument("--rerank-top", type=int, default=20)
    a = ap.parse_args()
    main(a.modes.split(","), a.collection, a.model, a.rerank, a.rerank_top)
