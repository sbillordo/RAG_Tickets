"""Seudonimización de texto de tickets antes de guardarlo o indexarlo.

- Personas: la misma persona recibe siempre el mismo marcador ([USUARIO_ab12cd]),
  así se conserva la señal "mismo usuario, caso parecido" sin guardar identidad.
- Secretos (contraseñas, tokens, claves): se redactan siempre ([SECRETO]).
- Identificadores (email, teléfono, DNI/NIE, IBAN): marcador estable por valor.
"""
import hashlib
import hmac
import os
import re
import unicodedata

from dotenv import load_dotenv

load_dotenv()

# Acepta prefijos (access_token, MessagePassword, db_pwd...) y valores entre comillas.
# No cruza saltos de línea y no vuelve a tocar valores ya redactados.
_SECRET_KV = re.compile(
    r"(?i)\b([\w-]*(?:contrase[ñn]a|password|passwd|pwd|secret|token|api[_-]?key)|clave|pass|secreto)\b"
    r"([ \t]*(?:es|is|:|=)[ \t]*)(?!\[SECRETO\])(\"[^\"\n]*\"|'[^'\n]*'|[^\s\"'<>&\]]+)"
)
_BEARER = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{12,}")
# Solo tokens sueltos (delimitados por espacios/comillas): así no se tocan parámetros de URL,
# adjuntos [cid:...] ni cabeceras MIME, que son ruido y no secretos.
_LONG_TOKEN = re.compile(
    r"(?<![\w/=&?:.<>\[%-])(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{32,}(?![\w/=&?:.<>\]%-])"
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,4})?\b")
_DNI_NIE = re.compile(r"\b(?:[XYZxyz]\d{7}|\d{8})[- ]?[A-Za-z]\b")
_PHONE = re.compile(r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\d[ .-]?){8,11}\d(?!\w)")

NAME_PART = re.compile(r"[^\W\d_]{3,}")  # palabras de 3+ letras: candidatas a parte de un nombre
_REPEATED_MARKER = re.compile(r"(\[USUARIO(?:_[0-9a-f]{6})?\])(?:[ \t]+\1)+")
_NEVER_NAMES = {"usuario", "email", "secreto"}  # palabras de los propios marcadores


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def name_parts(name: str | None) -> list[str]:
    """Partes de un nombre (en minúsculas) que podrían aparecer sueltas en un texto."""
    return [p.lower() for p in NAME_PART.findall(name or "")]


class Pseudonymizer:
    def __init__(self, users: dict[int, dict] | None = None, key: str | None = None,
                 exclude_parts: set[str] | None = None):
        """exclude_parts: palabras que NO deben tratarse como nombre aunque estén en el nombre de un
        usuario (organizaciones, buzones compartidos, palabras corrientes). Ver redact_pass.py."""
        key = key or os.getenv("PSEUDO_KEY")
        if not key:
            raise RuntimeError(
                "Falta PSEUDO_KEY en .env (secreto para generar los marcadores). "
                "Genera uno con: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        self._key = key.encode()
        exclude = {_strip_accents(p) for p in (exclude_parts or set())} | set(exclude_parts or ())
        exclude |= _NEVER_NAMES

        self._emails: dict[str, str] = {}   # email exacto -> marcador de su persona
        self._parts: dict[str, str] = {}    # parte de nombre -> marcador
        owners: dict[str, set[str]] = {}
        for user_id, u in (users or {}).items():
            marker = self.user_marker(user_id)
            if u.get("email"):
                self._emails[u["email"].strip().lower()] = marker
            for part in name_parts(u.get("name")):
                for variant in {part, _strip_accents(part)}:
                    if variant not in exclude:
                        owners.setdefault(variant, set()).add(marker)
        for part, ms in owners.items():
            # si varias personas comparten la palabra no se sabe cuál es: marcador genérico
            self._parts[part] = next(iter(ms)) if len(ms) == 1 else "[USUARIO]"

    def _digest(self, kind: str, value: str) -> str:
        return hmac.new(self._key, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()[:6]

    def _marker(self, kind: str, value: str) -> str:
        return f"[{kind}_{self._digest(kind, value.strip().lower())}]"

    def user_marker(self, user_id) -> str:
        return f"[USUARIO_{self._digest('USUARIO', str(user_id))}]"

    @staticmethod
    def redact_secrets(text: str) -> str:
        """Solo secretos. Es idempotente: se puede reaplicar sobre texto ya seudonimizado."""
        text = _SECRET_KV.sub(lambda m: f"{m.group(1)}{m.group(2)}[SECRETO]", text)
        text = _BEARER.sub(lambda m: f"{m.group(1)} [SECRETO]", text)
        return _LONG_TOKEN.sub("[SECRETO]", text)

    def redact_people(self, text: str) -> str:
        """Solo personas conocidas, palabra a palabra. Reaplicable sobre texto ya seudonimizado."""
        if not self._parts:
            return text
        text = NAME_PART.sub(lambda m: self._parts.get(m.group(0).lower(), m.group(0)), text)
        return _REPEATED_MARKER.sub(r"\1", text)  # "Ana García" -> un único marcador

    def text(self, text: str | None) -> str:
        if not text:
            return ""
        # 1) secretos primero, antes de que otros patrones los toquen
        text = self.redact_secrets(text)
        # 2) emails: de un usuario conocido -> su marcador; el resto -> marcador de email
        text = _EMAIL.sub(
            lambda m: self._emails.get(m.group(0).lower()) or self._marker("EMAIL", m.group(0)), text
        )
        # 3) personas conocidas (nombre y apellidos sueltos)
        text = self.redact_people(text)
        # 4) identificadores sueltos
        text = _IBAN.sub(lambda m: self._marker("IBAN", m.group(0).replace(" ", "")), text)
        text = _DNI_NIE.sub(lambda m: self._marker("DOC", re.sub(r"[- ]", "", m.group(0))), text)
        text = _PHONE.sub(lambda m: self._marker("TEL", re.sub(r"\D", "", m.group(0))), text)
        return text


if __name__ == "__main__":
    p = Pseudonymizer(
        users={1: {"name": "Ana García", "email": "ana.garcia@empresa.com"},
               2: {"name": "Grandvalira Resorts Andorra", "email": "info@gv.com"}},
        key="clave-de-prueba",
        exclude_parts={"grandvalira", "resorts", "andorra"},
    )
    demo = (
        "Hola, soy Ana García (ana.garcia@empresa.com, 600 123 456). "
        "Mi contraseña es Hunter2! y el DNI 12345678Z. IBAN ES9121000418450200051332. "
        "Otro: pepe@otra.com. Recuerda ana garcia. Grandvalira Resorts Andorra, temporada."
    )
    print(p.text(demo))
