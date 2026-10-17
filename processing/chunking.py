"""Trocea un ticket limpio en fragmentos que caben en el modelo de embeddings (~512 tokens).

Cada fragmento repite la cabecera (Asunto + Tags) para que tenga contexto por sí solo.
"""
import re

CHUNK_CHARS = 1200    # cuerpo de cada fragmento (+ cabecera ≈ 1.400 caracteres ≈ 400 tokens)
OVERLAP_CHARS = 150   # solapamiento entre fragmentos consecutivos


def _units(body: str) -> list[str]:
    """Divide en frases/líneas; las frases gigantes se parten a la fuerza."""
    units = []
    for line in body.split("\n"):
        for sent in re.split(r"(?<=[.!?])\s+", line):
            while len(sent) > CHUNK_CHARS:
                units.append(sent[:CHUNK_CHARS])
                sent = sent[CHUNK_CHARS:]
            if sent.strip():
                units.append(sent)
    return units


def chunk_ticket(clean_text: str) -> list[str]:
    lines = clean_text.split("\n", 2)
    header = "\n".join(lines[:2]) if len(lines) >= 2 else clean_text  # Asunto + Tags
    body = lines[2] if len(lines) > 2 else ""
    if len(header) + len(body) <= CHUNK_CHARS + 200:
        return [clean_text]

    chunks, current, size = [], [], 0
    for u in _units(body):
        if size + len(u) > CHUNK_CHARS and current:
            chunks.append(current)
            tail, tsize = [], 0
            for prev in reversed(current):  # solapamiento: últimas frases del fragmento anterior
                if tsize + len(prev) > OVERLAP_CHARS:
                    break
                tail.insert(0, prev)
                tsize += len(prev) + 1
            current, size = tail, tsize
        current.append(u)
        size += len(u) + 1
    if current:
        chunks.append(current)
    return [f"{header}\n" + "\n".join(c) for c in chunks]


# ---------- documentación (Notion): troceado por secciones ----------
_HEADING = re.compile(r"^(#{1,3}) +(.+)$")
MIN_SECTION_CHARS = 400   # una sección ya "llena" no se mezcla con la siguiente


def chunk_markdown(content: str) -> list[str]:
    """Trocea una página de documentación respetando sus títulos.

    Las 3 primeras líneas (Título, Categorías, Equipo|Tipo|Estado) se repiten en cada fragmento,
    junto con la sección ("Sección: Título › Subtítulo") para que cada trozo tenga contexto propio.
    """
    lines = content.split("\n")
    head, body = "\n".join(lines[:3]), lines[3:]
    if len(content) <= CHUNK_CHARS + 200:
        return [content]

    # los títulos se jerarquizan respecto al nivel más alto que use la página (muchas solo usan ##)
    min_level = min((len(m.group(1)) for l in body if (m := _HEADING.match(l))), default=1)
    path: list[str] = []
    out: list[tuple[str, list[str]]] = []   # (sección, líneas)
    cur: list[str] = []
    size, label = 0, ""

    def flush(carry: bool):
        nonlocal cur, size
        if any(l.strip() for l in cur):
            out.append((label, cur))
        tail, tsize = [], 0
        if carry:  # solapamiento solo cuando el corte es forzado por tamaño, no en un título
            for prev in reversed(cur):
                if tsize + len(prev) > OVERLAP_CHARS:
                    break
                tail.insert(0, prev)
                tsize += len(prev) + 1
        cur, size = tail, tsize

    for line in body:
        m = _HEADING.match(line)
        if m:
            level = len(m.group(1)) - min_level + 1
            path = path[: level - 1] + [m.group(2).strip()]
            if size >= MIN_SECTION_CHARS:
                flush(carry=False)
            label = " › ".join(path)
        for piece in ([line] if len(line) <= CHUNK_CHARS else _units(line)):
            if size + len(piece) > CHUNK_CHARS and cur:
                flush(carry=True)
            cur.append(piece)
            size += len(piece) + 1
    flush(carry=False)

    return [
        f"{head}\n" + (f"Sección: {sec}\n" if sec else "") + "\n".join(ls).strip()
        for sec, ls in out
    ]
