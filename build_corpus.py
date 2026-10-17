"""Fase 1: Zendesk -> seudonimización (en memoria) -> data/zendesk_docs.jsonl

Uso:
    python build_corpus.py --limit 50     # prueba con pocos tickets
    python build_corpus.py                # todos los solved/closed
"""
import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from ingestion.zendesk import iter_closed_tickets
from models.models import Document
from processing.pseudonymizer import Pseudonymizer

OUT_PATH = Path("data/zendesk_docs.jsonl")
LOG_PATH = Path("logs/build_corpus.log")


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_document(t: dict, pseudo: Pseudonymizer) -> Document:
    parts = [
        f"Asunto: {pseudo.text(t['subject'])}",
        f"Tags: {', '.join(t['tags'])}" if t["tags"] else "Tags: (ninguno)",
        f"Descripción: {pseudo.text(t['description'])}",
    ]
    if t["comments"]:
        parts.append("Conversación:")
        for c in t["comments"]:
            kind = "" if c["public"] else " (nota interna)"
            parts.append(f"- {pseudo.user_marker(c['author_id'])}{kind}: {pseudo.text(c['body'])}")

    return Document(
        id=f"zendesk-{t['id']}",
        source="zendesk",
        status=t["status"],
        title=pseudo.text(t["subject"]),
        content="\n".join(parts),
        created_at=_dt(t["created_at"]),
        updated_at=_dt(t["updated_at"]),
        metadata={
            "ticket_id": t["id"],
            "url": t["url"],
            "tags": t["tags"],
            "requester": pseudo.user_marker(t["requester_id"]),
            "assignee_name": t["assignee_name"],
            "pendiente_de": t["pendiente_de"],
            "id_externo": t["id_externo"],
        },
    )


def _setup_log() -> logging.Logger:
    LOG_PATH.parent.mkdir(exist_ok=True)
    log = logging.getLogger("build_corpus")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log


def main(limit: int | None):
    log = _setup_log()
    started = time.monotonic()
    log.info("INICIO  limit=%s", limit)
    n = 0
    try:
        stream = iter_closed_tickets(limit=limit)
        users = next(stream)  # el primer elemento son los usuarios
        pseudo = Pseudonymizer(users=users)

        OUT_PATH.parent.mkdir(exist_ok=True)
        with OUT_PATH.open("w", encoding="utf-8") as f:
            for t in stream:
                doc = build_document(t, pseudo)
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
                n += 1
                if n % 25 == 0:
                    log.info("%d tickets procesados", n)
    except BaseException as e:
        log.error("ERROR tras %d tickets (%s): %s", n, type(e).__name__, e)
        raise
    finally:
        secs = int(time.monotonic() - started)
        log.info("FIN  %d documentos en %s  duración=%dh %02dm %02ds",
                 n, OUT_PATH, secs // 3600, secs % 3600 // 60, secs % 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    main(ap.parse_args().limit)
