"""Interfaz web local del buscador (tickets de Zendesk + documentación de Notion).

Uso:  python app.py            -> abre http://127.0.0.1:8765 en el navegador
      python app.py --no-browser

Solo escucha en 127.0.0.1: no es accesible desde otros equipos de la red.
Sin dependencias nuevas: servidor HTTP de la librería estándar de Python.
"""
import argparse
import json
import threading
import traceback
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import search as S

HOST = "127.0.0.1"
UI_FILE = Path(__file__).parent / "ui" / "index.html"
_search_lock = threading.Lock()  # una búsqueda cada vez: el modelo ya usa todos los núcleos


def _run_search(body: dict) -> dict:
    ticket = int(body["ticket"]) if str(body.get("ticket") or "").strip() else None
    texto = (body.get("texto") or "").strip()
    if ticket:
        texto = S.ticket_text(ticket)
    if not texto:
        raise ValueError("Escribe el texto del caso o indica un número de ticket.")
    out = S.search(
        texto,
        k=max(1, min(int(body.get("k") or 5), 20)),
        kdocs=max(0, min(int(body.get("kdocs") or 3), 10)),
        tag=(body.get("tag") or "").strip() or None,
        desde=int(body["desde"]) if str(body.get("desde") or "").strip() else None,
        solo_finalizadas=bool(body.get("solo_finalizadas")),
        excluir_ticket=ticket,
        modo="semantico" if body.get("modo") == "semantico" else "hibrido",
    )
    tickets = [{
        "id": t["id"],
        "title": t["title"],
        "status": t["status"],
        "tags": list(t["tags"]),
        "date": datetime.fromtimestamp(t["created_ts"]).strftime("%Y-%m-%d"),
        "rank_sem": t["rank_sem"],
        "cos": round(float(t["cos"]), 3) if t["cos"] is not None else None,
        "rank_pal": t["rank_pal"],
        "url": S.ticket_link(t["id"]),
    } for t in out["tickets"]]
    docs = [{
        "score": round(float(score), 3),
        "doc_ref": m.get("doc_ref", ""),
        "title": m["title"],
        "estado": m.get("estado") or "sin estado",
        "tags": m.get("tags", ""),
        "section": next((l[len("Sección: "):] for l in doc.split("\n") if l.startswith("Sección: ")), ""),
        "url": m.get("url", ""),
    } for score, m, doc in out["docs"]]
    return {"tickets": tickets, "docs": docs, "hibrido": out["hibrido"], "query_ticket": ticket}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, data: dict):
        self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, UI_FILE.read_bytes(), "text/html; charset=utf-8")
        else:
            self._json(404, {"error": "no encontrado"})

    def do_POST(self):
        if self.path != "/api/search":
            return self._json(404, {"error": "no encontrado"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            with _search_lock:
                self._json(200, _run_search(body))
        except ValueError as e:
            self._json(400, {"error": str(e)})
        except Exception as e:  # el detalle va a la consola, no al navegador
            traceback.print_exc()
            self._json(500, {"error": f"Error interno ({type(e).__name__}). Revisa la consola del servidor."})

    def log_message(self, fmt, *args):  # sin registrar cada petición (podrían contener texto de tickets)
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    print("Cargando modelo, índices y usuarios de Zendesk (unos segundos)...", flush=True)
    S.warmup()
    url = f"http://{HOST}:{a.port}"
    server = ThreadingHTTPServer((HOST, a.port), Handler)
    print(f"Listo: {url}   (Ctrl+C para parar)", flush=True)
    if not a.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nParado.")


if __name__ == "__main__":
    main()
