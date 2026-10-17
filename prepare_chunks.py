"""Fase 2a: documentos seudonimizados -> limpieza -> chunking -> fragmentos JSONL

Uso:
    python prepare_chunks.py                  # Zendesk: data/zendesk_docs.jsonl -> data/chunks.jsonl
    python prepare_chunks.py --source notion  # Notion:  data/notion_docs.jsonl  -> data/notion_chunks.jsonl
"""
import argparse
import json
import re
from pathlib import Path

from processing.chunking import chunk_markdown, chunk_ticket
from processing.cleaning import clean_ticket

SOURCES = {
    "zendesk": (Path("data/zendesk_docs.jsonl"), Path("data/chunks.jsonl")),
    "notion": (Path("data/notion_docs.jsonl"), Path("data/notion_chunks.jsonl")),
}


def _tidy(text: str) -> str:
    """Limpieza mínima para documentación: sin los avisos/firmas de correo propios de los tickets."""
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main(source: str):
    src_path, dst_path = SOURCES[source]
    n_docs = n_chunks = chars_in = chars_out = 0
    with src_path.open(encoding="utf-8") as src, dst_path.open("w", encoding="utf-8") as dst:
        for line in src:
            d = json.loads(line)
            if source == "zendesk":
                clean = clean_ticket(d["content"])
                pieces = chunk_ticket(clean)
            else:
                clean = _tidy(d["content"])
                pieces = chunk_markdown(clean)
            chars_in += len(d["content"])
            chars_out += len(clean)
            for i, text in enumerate(pieces):
                dst.write(json.dumps({
                    "chunk_id": f"{d['id']}#{i}",
                    "doc_id": d["id"],
                    "text": text,
                    "title": d["title"],
                    "status": d["status"],
                    "created_at": d["created_at"],
                    "updated_at": d["updated_at"],
                    "metadata": d["metadata"],
                }, ensure_ascii=False) + "\n")
                n_chunks += 1
            n_docs += 1
    print(f"[{source}] {n_docs} documentos -> {n_chunks} fragmentos ({n_chunks / n_docs:.1f} por documento)")
    print(f"texto: {chars_in:,} -> {chars_out:,} caracteres ({100 * (1 - chars_out / chars_in):.0f} % menos)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=SOURCES, default="zendesk")
    main(ap.parse_args().source)
