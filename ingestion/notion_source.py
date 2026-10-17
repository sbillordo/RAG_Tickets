"""Ingesta de la base de datos 'Documentación' de Notion (solo lectura).

Solo consulta la base NOTION_DB_ID; no recorre nada más del workspace.
Devuelve texto SIN tratar: no escribir a disco sin pasar antes por Pseudonymizer.
"""
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
DB_ID = os.getenv("NOTION_DB_ID", "c341125a-f0f9-407f-9b43-94d600f73bd1")  # base 'Documentación'
MIN_INTERVAL = 0.35   # Notion permite ~3 peticiones/s de media
MAX_DEPTH = 6

_last_call = 0.0


def _request(method: str, path: str, **kwargs) -> dict:
    global _last_call
    headers = {
        "Authorization": f"Bearer {os.getenv('NOTION_TOKEN')}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    for attempt in range(6):
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        r = requests.request(method, f"{API}{path}", headers=headers, timeout=60, **kwargs)
        if r.status_code == 429:  # límite de peticiones: esperar lo que indique Notion
            time.sleep(float(r.headers.get("Retry-After", 2)) + 0.5)
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Notion no responde correctamente: {method} {path}")


def _paginate(method: str, path: str, body: dict | None = None):
    cursor = None
    while True:
        params = {"page_size": 100}
        if method == "GET":
            if cursor:
                params["start_cursor"] = cursor
            data = _request("GET", path, params=params)
        else:
            data = _request("POST", path, json={**(body or {}), **params, **({"start_cursor": cursor} if cursor else {})})
        yield from data["results"]
        if not data.get("has_more"):
            return
        cursor = data["next_cursor"]


# ---------- propiedades ----------
def _plain(rich: list) -> str:
    return "".join(t.get("plain_text", "") for t in rich or [])


def _prop(page: dict, name: str):
    p = page["properties"].get(name)
    if not p:
        return None
    t = p["type"]
    if t == "title" or t == "rich_text":
        return _plain(p[t])
    if t in ("select", "status"):
        return (p[t] or {}).get("name")
    if t == "multi_select":
        return [o["name"] for o in p[t]]
    if t == "unique_id":
        u = p["unique_id"]
        return f"{u.get('prefix') or ''}-{u['number']}" if u.get("prefix") else str(u["number"])
    return None


def _title(page: dict) -> str:
    for p in page["properties"].values():
        if p["type"] == "title":
            return _plain(p["title"]).strip()
    return ""


# ---------- bloques -> texto ----------
_TEXT_BLOCKS = {
    "paragraph": "", "heading_1": "# ", "heading_2": "## ", "heading_3": "### ",
    "bulleted_list_item": "- ", "numbered_list_item": "1. ", "to_do": "- [ ] ",
    "toggle": "", "quote": "> ", "callout": "", "code": "",
}
_MEDIA = {"image", "file", "pdf", "video", "audio"}


def _block_lines(block: dict, depth: int, stats: dict) -> list[str]:
    t = block["type"]
    data = block[t]
    indent = "  " * depth
    lines: list[str] = []
    if t in _TEXT_BLOCKS:
        text = _plain(data.get("rich_text"))
        if t == "code":
            text = f"```{data.get('language', '')}\n{text}\n```"
        if text.strip():
            lines.append(f"{indent}{_TEXT_BLOCKS[t]}{text}")
    elif t in ("bookmark", "embed", "link_preview"):
        lines.append(f"{indent}[enlace: {data.get('url', '')}]")
    elif t in _MEDIA:
        stats["media"] += 1
        caption = _plain(data.get("caption"))
        lines.append(f"{indent}[{t} omitido{': ' + caption if caption else ''}]")
    elif t == "equation":
        lines.append(f"{indent}{data.get('expression', '')}")
    elif t == "table_row":
        lines.append(f"{indent}| " + " | ".join(_plain(c) for c in data["cells"]) + " |")
    elif t == "child_page":
        lines.append(f"{indent}[subpágina: {data.get('title', '')}]")
        return lines  # el contenido de la subpágina no se incluye
    elif t == "child_database":
        lines.append(f"{indent}[base de datos incrustada: {data.get('title', '')}]")
        return lines
    # (divider, breadcrumb, table_of_contents, etc.: sin texto)

    if block.get("has_children") and depth < MAX_DEPTH:
        child_depth = depth if t in ("column_list", "column", "synced_block", "table") else depth + 1
        for child in _paginate("GET", f"/blocks/{block['id']}/children"):
            lines.extend(_block_lines(child, child_depth, stats))
    return lines


def page_body(page_id: str) -> tuple[str, dict]:
    stats = {"media": 0}
    lines: list[str] = []
    for block in _paginate("GET", f"/blocks/{page_id}/children"):
        lines.extend(_block_lines(block, 0, stats))
    return "\n".join(lines), stats


# ---------- API pública ----------
def iter_pages(excluir_estados: set[str] | None = None, limit: int | None = None):
    """Genera las páginas de la base con propiedades y cuerpo (texto sin tratar)."""
    excluir = excluir_estados if excluir_estados is not None else {"99.DISCARD"}
    n = 0
    for page in _paginate("POST", f"/databases/{DB_ID}/query"):
        if page.get("archived") or page.get("in_trash"):
            continue
        estado = _prop(page, "Estado")
        if estado in excluir:
            continue
        body, stats = page_body(page["id"])
        yield {
            "page_id": page["id"],
            "doc_id": _prop(page, "ID"),
            "title": _title(page),
            "url": page.get("url"),
            "estado": estado,
            "equipo": _prop(page, "Equipo"),
            "type": _prop(page, "Type") or [],
            "categorias": _prop(page, "Categorías") or [],
            "nivel_impacto": _prop(page, "Nivel de Impacto"),
            "descripcion": _prop(page, "Descripción") or "",
            "created_time": page["created_time"],
            "last_edited_time": page["last_edited_time"],
            "body": body,
            "media_omitidos": stats["media"],
            # Responsable / Colaboradores / Created by: NO se leen, son nombres de personas.
        }
        n += 1
        if limit and n >= limit:
            return
