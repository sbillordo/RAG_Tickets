"""Construye el índice de palabras (BM25) de los tickets a partir de data/chunks.jsonl.

Uso:  python build_bm25.py        (tarda ~1 min; repetir cada vez que se regeneren los fragmentos)
Genera data/bm25_zendesk.pkl, que usa search.py para la búsqueda híbrida.
"""
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from processing.sparse import Bm25

SRC = Path("data/chunks.jsonl")
DST = Path("data/bm25_zendesk.pkl")


def main():
    t0 = time.monotonic()
    chunks = [json.loads(l) for l in SRC.open(encoding="utf-8")]
    tickets = {}
    for c in chunks:
        m = c["metadata"]
        tid = int(m["ticket_id"])
        if tid not in tickets:
            created = datetime.fromisoformat(c["created_at"])
            tickets[tid] = {
                "title": c["title"], "status": c["status"], "tags": m.get("tags") or [],
                "created_ts": int(created.timestamp()), "year": created.year,
            }
    bm25 = Bm25().fit([c["text"] for c in chunks])
    data = {
        "bm25": bm25,
        "ticket_id": np.array([int(c["metadata"]["ticket_id"]) for c in chunks]),
        "tags": [set(c["metadata"].get("tags") or []) for c in chunks],
        "year": np.array([datetime.fromisoformat(c["created_at"]).year for c in chunks]),
        "tickets": tickets,
    }
    with DST.open("wb") as f:
        pickle.dump(data, f)
    print(f"{len(chunks)} fragmentos / {len(tickets)} tickets -> {DST} "
          f"({DST.stat().st_size / 1e6:.0f} MB) en {time.monotonic() - t0:.0f} s")


if __name__ == "__main__":
    main()
