"""Prueba de OCR sobre una muestra de imágenes de la documentación de Notion.

Uso:  python ocr_sample.py [--n 20] [--seed 7]

- Las imágenes se descargan EN MEMORIA y no se guardan en disco.
- El texto extraído pasa por la misma seudonimización que el resto del corpus antes de escribirse.
- Resultado: data/ocr_muestra.md, para validar a mano comparando con la imagen en Notion.
"""
import argparse
import io
import json
import random
import statistics as st
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from ingestion.notion_source import _paginate
from ingestion.zendesk import get_users_zendesk
from processing.pseudonymizer import Pseudonymizer
from redact_pass import load_exclude

DOCS = Path("data/notion_docs.jsonl")
OUT = Path("data/ocr_muestra.md")
MIN_CONF = 0.25          # fragmentos con menos confianza se descartan
MAX_PER_PAGE = 2         # variedad: como mucho 2 imágenes por página


def image_blocks(block_id: str, depth: int = 0):
    """Bloques de imagen de una página, en orden, incluidos los anidados (columnas, toggles...)."""
    for b in _paginate("GET", f"/blocks/{block_id}/children"):
        if b["type"] == "image":
            yield b
        if b.get("has_children") and depth < 4 and b["type"] not in ("child_page", "child_database"):
            yield from image_blocks(b["id"], depth + 1)


def image_url(block: dict) -> str:
    img = block["image"]
    return img[img["type"]]["url"]  # 'file' (enlace temporal de Notion) o 'external'


def ocr_lines(reader, img: np.ndarray):
    """OCR y agrupación de los fragmentos en líneas (de arriba abajo, de izquierda a derecha)."""
    items = [(box, text, conf) for box, text, conf in reader.readtext(img, detail=1) if conf >= MIN_CONF]
    if not items:
        return [], []
    heights = [box[2][1] - box[0][1] for box, _, _ in items]
    tol = max(6.0, st.median(heights) * 0.6)
    items.sort(key=lambda it: ((it[0][0][1] + it[0][2][1]) / 2, it[0][0][0]))
    lines, current, cur_y = [], [], None
    for box, text, conf in items:
        y = (box[0][1] + box[2][1]) / 2
        if current and abs(y - cur_y) > tol:  # otra línea: se cierra la actual
            lines.append(" ".join(t for _, t in sorted(current)))
            current = []
        if not current:
            cur_y = y  # altura de referencia de la línea nueva
        current.append((box[0][0], text))
    if current:
        lines.append(" ".join(t for _, t in sorted(current)))
    return lines, [conf for _, _, conf in items]


def main(n: int, seed: int):
    import easyocr  # import aquí: tarda en cargar

    docs = [json.loads(l) for l in DOCS.open(encoding="utf-8")]
    with_images = [d for d in docs if "[image omitido" in d["content"]]
    random.Random(seed).shuffle(with_images)
    print(f"páginas con imágenes: {len(with_images)}; se toma una muestra de {n} imágenes")

    users = get_users_zendesk()
    pseudo = Pseudonymizer(users=users, exclude_parts=load_exclude(users))
    print("cargando EasyOCR (castellano e inglés; el catalán comparte alfabeto)...", flush=True)
    reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)

    results = []
    for d in with_images:
        if len(results) >= n:
            break
        blocks = list(image_blocks(d["metadata"]["page_id"]))
        for pos, b in enumerate(blocks, start=1):
            if len(results) >= n or sum(r["doc"] == d["id"] for r in results) >= MAX_PER_PAGE:
                break
            try:
                raw = requests.get(image_url(b), timeout=60).content          # solo en memoria
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as e:
                print(f"  {d['metadata']['doc_ref']} imagen {pos}: no se pudo leer ({type(e).__name__})")
                continue
            t0 = time.monotonic()
            lines, confs = ocr_lines(reader, np.array(img))
            secs = time.monotonic() - t0
            text = pseudo.text("\n".join(lines))                                # seudonimizado antes de guardar
            caption = "".join(t.get("plain_text", "") for t in b["image"].get("caption") or [])
            results.append({
                "doc": d["id"], "ref": d["metadata"]["doc_ref"], "title": d["title"], "url": d["metadata"]["url"],
                "pos": pos, "total": len(blocks), "caption": pseudo.text(caption), "size": img.size,
                "secs": secs, "conf": st.mean(confs) if confs else None, "text": text,
                "secretos": text.count("[SECRETO]"),
            })
            print(f"  [{len(results):2d}/{n}] {d['metadata']['doc_ref']} imagen {pos}/{len(blocks)}: "
                  f"{len(text):5d} caracteres en {secs:4.1f} s", flush=True)
            del raw, img

    write_report(results)
    print(f"\nInforme: {OUT}")


def write_report(results):
    secs = [r["secs"] for r in results]
    with_text = [r for r in results if r["text"].strip()]
    out = [
        "# Prueba de OCR sobre imágenes de Notion",
        "",
        f"- Imágenes procesadas: **{len(results)}** · con texto detectado: **{len(with_text)}**",
        f"- Tiempo por imagen: mediana **{st.median(secs):.1f} s**, máximo {max(secs):.1f} s (CPU)" if secs else "",
        f"- Estimación para las ~2.541 imágenes de Notion: **~{st.median(secs) * 2541 / 3600:.1f} h** (solo OCR)" if secs else "",
        "- El texto está **seudonimizado**. Las imágenes no se han guardado.",
        "",
        "**Cómo validar:** abre la página en Notion, busca la imagen nº indicado (contando desde arriba) "
        "y comprueba si el texto extraído recoge lo importante (errores, menús, campos).",
        "",
    ]
    for i, r in enumerate(results, start=1):
        conf = f"{r['conf']:.2f}" if r["conf"] is not None else "-"
        out += [
            f"## {i}. {r['ref']} · {r['title']}",
            "",
            f"- Página: {r['url']}",
            f"- Imagen nº **{r['pos']}** de {r['total']} en la página"
            + (f" · pie de foto: «{r['caption']}»" if r["caption"] else ""),
            f"- Tamaño {r['size'][0]}×{r['size'][1]} px · {r['secs']:.1f} s · confianza media {conf}"
            + (f" · **{r['secretos']} secreto(s) tapado(s)**" if r["secretos"] else ""),
            "",
            "```text",
            r["text"].strip() or "(no se detectó texto)",
            "```",
            "",
            "Valoración (rellenar): ☐ útil  ☐ parcial  ☐ inútil — comentarios:",
            "",
        ]
    OUT.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main(a.n, a.seed)
