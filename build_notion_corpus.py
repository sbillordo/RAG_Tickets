"""Notion (base 'Documentación') -> seudonimización en memoria -> data/notion_docs.jsonl

Uso:
    python build_notion_corpus.py --limit 5      # prueba
    python build_notion_corpus.py                # todas las páginas excepto 99.DISCARD

Además de los documentos escribe logs/notion_review.log: por página, cuántos secretos/emails/
teléfonos/etc. se han tapado (solo recuentos y enlace, nunca contenido), para limpiar el origen.
"""
import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from ingestion.notion_source import iter_pages
from ingestion.zendesk import get_users_zendesk
from models.models import Document
from processing.pseudonymizer import Pseudonymizer
from redact_pass import load_exclude

OUT_PATH = Path("data/notion_docs.jsonl")
LOG_PATH = Path("logs/build_notion.log")
REVIEW_PATH = Path("logs/notion_review.log")
MARKERS = {"secretos": "[SECRETO]", "emails": "[EMAIL_", "telefonos": "[TEL_", "documentos": "[DOC_", "iban": "[IBAN_"}


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_document(p: dict, pseudo: Pseudonymizer) -> tuple[Document, dict]:
    head = [
        f"Título: {pseudo.text(p['title'])}",
        f"Categorías: {', '.join(p['categorias'])}" if p["categorias"] else "Categorías: (ninguna)",
        f"Equipo: {p['equipo'] or '-'} | Tipo: {', '.join(p['type']) or '-'} | Estado: {p['estado'] or '-'}",
    ]
    if p["descripcion"].strip():
        head.append(f"Descripción: {pseudo.text(p['descripcion'])}")
    content = "\n".join(head) + "\n\n" + pseudo.text(p["body"])
    found = {k: content.count(m) for k, m in MARKERS.items()}
    found["media"] = p["media_omitidos"]

    doc = Document(
        id=f"notion-{p['doc_id'] or p['page_id']}",
        source="notion",
        status=p["estado"] or "sin_estado",
        title=pseudo.text(p["title"]),
        content=content,
        created_at=_dt(p["created_time"]),
        updated_at=_dt(p["last_edited_time"]),
        metadata={
            "doc_ref": p["doc_id"] or "",
            "page_id": p["page_id"],
            "url": p["url"],
            "tags": p["categorias"],          # las categorías hacen de tags (filtro tag__<nombre>)
            "equipo": p["equipo"] or "",
            "tipo": ", ".join(p["type"]),
            "estado": p["estado"] or "",
            "nivel_impacto": p["nivel_impacto"] or "",
        },
    )
    return doc, found


def main(limit: int | None):
    LOG_PATH.parent.mkdir(exist_ok=True)
    log = logging.getLogger("notion")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)

    t0 = time.monotonic()
    log.info("INICIO  limit=%s", limit)
    users = get_users_zendesk()
    pseudo = Pseudonymizer(users=users, exclude_parts=load_exclude(users))

    n = flagged = 0
    OUT_PATH.parent.mkdir(exist_ok=True)
    try:
        with OUT_PATH.open("w", encoding="utf-8") as out, REVIEW_PATH.open("w", encoding="utf-8") as review:
            review.write("Páginas con datos sensibles tapados (revisar y limpiar en Notion). Solo recuentos.\n")
            for page in iter_pages(limit=limit):
                doc, found = build_document(page, pseudo)
                out.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
                n += 1
                if any(v for k, v in found.items() if k != "media") or found["media"]:
                    flagged += 1
                    parts = " ".join(f"{k}={v}" for k, v in found.items() if v)
                    review.write(f"{page['doc_id'] or page['page_id']} | {parts} | {page['url']}\n")
                if n % 20 == 0:
                    log.info("%d páginas procesadas", n)
    except BaseException as e:
        log.error("ERROR tras %d páginas (%s): %s", n, type(e).__name__, e)
        raise
    finally:
        secs = int(time.monotonic() - t0)
        log.info("FIN  %d documentos en %s (%d con avisos en %s)  duración=%dm %02ds",
                 n, OUT_PATH, flagged, REVIEW_PATH, secs // 60, secs % 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    main(ap.parse_args().limit)
