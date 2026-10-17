"""Limpieza de ruido de los tickets (sobre texto ya seudonimizado) antes de trocear.

Quita lo que hace que dos tickets se parezcan por el "pie del correo" y no por el problema:
avisos legales, correos citados, adjuntos [cid:], marcadores sueltos, duplicados, URLs largas...
"""
import html
import re
from dataclasses import dataclass, field

COMMENT_RE = re.compile(r"^- (\[USUARIO(?:_[0-9a-f]{6})?\])( \(nota interna\))?: ", re.M)

_LEGAL = re.compile(
    r"(?is)\bAv[ií]s\s+(?:de\s+Confidencialit\w*|Legal)\b.{0,250}?Notice\)?(?:\s*[(<]?https?://\S+[)>]?)?"
)
_QUOTE_STARTS = [
    re.compile(r"(?im)^\s*-{2,}\s*(?:original message|mensaje original|mensaje reenviado|forwarded message)\b"),
    re.compile(r"(?im)^\s*(?:De|From|Von)\s*:.*\n(?:.*\n){0,4}?\s*(?:Enviado(?: el)?|Sent|Date|Fecha|Para|To|Asunto|Subject)\s*:"),
    re.compile(r"(?im)^\s*El .{5,120}(?:escribi[óo]|va escriure)\s*:\s*$"),
    re.compile(r"(?im)^\s*On .{5,120} wrote\s*:\s*$"),
]
_CID = re.compile(r"\[cid:[^\]]*\]+")
_MAILTO = re.compile(r"<?mailto:\S*?>?(?=\s|$)")
_URL = re.compile(r"\(?<?https?://([^\s/)>]+)\S*?(?:[)>]|(?=\s|$))")
_ONLY_MARKERS = re.compile(r"^[\W_]*(?:\[(?:EMAIL|TEL|USUARIO)(?:_\w+)?\][\W_]*)*$")
_COMPANY_LINE = re.compile(r"(?im)^\s*Grandvalira Resorts(?: Andorra)?\s*$")

# Notas automáticas de Zendesk al fusionar tickets. En el destino copian asunto y último comentario del
# origen ("La solicitud #123 ... fue cerrada y fusionada en esta solicitud. Último comentario..."):
# son texto de OTRO caso y contaminan el índice (y cualquier evaluación).
_MERGE_NOTICE = re.compile(
    r"(?is)\b(?:fusionad[ao]|fusionat\w*|combinad[ao]|merged)\b.{0,80}?\b(?:solicitud|request|sol[·.]?licitud)\b"
)

MAX_LINE = 400
MAX_BODY = 3500

# --- limpieza adicional SOLO para la consulta (un caso nuevo) ---
_CLOSING = re.compile(
    r"(?im)^\s*(?:muchas gracias|gracias|gràcies|moltes gràcies|merci|thanks|thank you|un saludo|saludos|"
    r"salutacions|cordialmente|cordialment|atentamente|atentament|best regards|kind regards|regards)\b"
)
_MARKERS = re.compile(r"\[(?:USUARIO|EMAIL|TEL|DOC|IBAN)(?:_\w+)?\]")


def strip_signature(text: str, min_keep: int = 60) -> str:
    """Corta en la primera despedida ("Gracias", "Saludos"...) si queda texto suficiente antes."""
    m = _CLOSING.search(text)
    return text[:m.start()].rstrip() if m and len(text[:m.start()].strip()) >= min_keep else text


def strip_markers(text: str) -> str:
    """Quita [USUARIO_x], [EMAIL_x]...: dejan de pesar en el significado (y la coincidencia por persona)."""
    return re.sub(r"[ \t]{2,}", " ", _MARKERS.sub("", text))


def _norm(s: str) -> str:
    return re.sub(r"\W+", "", s.lower())


def _cut_quoted(text: str) -> str:
    cut = len(text)
    for rx in _QUOTE_STARTS:
        m = rx.search(text)
        if m:
            cut = min(cut, m.start())
    # si el correo es solo un reenvío (casi nada antes de la cita), se conserva entero
    return text[:cut] if len(text[:cut].strip()) >= 40 else text


def clean_body(text: str) -> str:
    text = html.unescape(text).replace("\xa0", " ").replace("\r", "")
    text = _cut_quoted(text)
    text = _LEGAL.sub(" ", text)
    text = _CID.sub(" ", text)
    text = _MAILTO.sub(" ", text)
    text = _URL.sub(lambda m: f" {m.group(1)} ", text)
    text = _COMPANY_LINE.sub("", text)
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line or _ONLY_MARKERS.match(line):
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.append(line if len(line) <= MAX_LINE else line[:MAX_LINE] + " […]")
    text = "\n".join(lines).strip()
    if len(text) > MAX_BODY:  # trazas / JSON enormes: cabeza y cola
        text = text[:2600] + "\n[…recortado…]\n" + text[-800:]
    return text


@dataclass
class ParsedTicket:
    subject: str
    tags: str
    description: str
    comments: list = field(default_factory=list)  # (autor, es_interna, cuerpo)


def parse_content(content: str) -> ParsedTicket:
    head, _, conv = content.partition("\nConversación:\n")
    subject = tags = description = ""
    m = re.match(r"Asunto: (.*)\nTags: (.*)\nDescripción: (.*)", head, re.S)
    if m:
        subject, tags, description = m.group(1), m.group(2), m.group(3)
    comments = []
    pieces = COMMENT_RE.split(conv) if conv else []
    # split con 2 grupos: [pre, autor, interna, cuerpo, autor, interna, cuerpo, ...]
    for i in range(1, len(pieces) - 2, 3):
        comments.append((pieces[i], bool(pieces[i + 1]), pieces[i + 2]))
    return ParsedTicket(subject, tags, description, comments)


def clean_ticket(content: str) -> str:
    """Devuelve el texto limpio y sin duplicados: cabecera + descripción + conversación."""
    p = parse_content(content)
    desc = clean_body(p.description)
    out = [f"Asunto: {p.subject}", f"Tags: {p.tags}", f"Descripción: {desc}"]
    seen = {_norm(desc)}
    conv = []
    for author, internal, body in p.comments:
        if _MERGE_NOTICE.search(body[:300]):
            continue  # nota automática de fusión de tickets
        body = clean_body(body)
        key = _norm(body)
        if not key or key in seen:  # vacío o repetido (p. ej. la descripción como 1.er comentario)
            continue
        seen.add(key)
        conv.append(f"- {author}{' (nota interna)' if internal else ''}: {body}")
    if conv:
        out.append("Conversación:")
        out.extend(conv)
    return "\n".join(out)
