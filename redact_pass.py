"""Reaplica seudonimización de secretos y personas sobre data/zendesk_docs.jsonl.

Útil al mejorar las reglas de Pseudonymizer sin repetir la extracción (1h20m).
Es idempotente: reaplicarlo sobre texto ya seudonimizado no lo altera.
Los marcadores de persona coinciden con los de la extracción (HMAC con la misma PSEUDO_KEY).

Qué palabras se tratan como nombre: las partes de nombre de los usuarios de Zendesk, MENOS las que
en el corpus se escriben casi siempre en minúscula (palabras corrientes) y las de ORG_WORDS.
"""
import json
import os
import re
from collections import Counter
from pathlib import Path

from ingestion.zendesk import get_users_zendesk
from processing.pseudonymizer import NAME_PART, Pseudonymizer, name_parts

PATH = Path("data/zendesk_docs.jsonl")
TMP = PATH.with_suffix(".tmp")
MIN_CAPITALIZED = 0.90  # un nombre propio casi siempre va con mayúscula inicial
MAX_UPPER = 0.50        # si se escribe en MAYÚSCULAS la mitad de las veces, es una sigla

# Organizaciones, buzones compartidos y topónimos que aparecen como "usuarios" en Zendesk.
ORG_WORDS = {
    "grandvalira", "andorra", "resorts", "nevasa", "ensisa", "vallnord", "ordino", "arcalis",
    "ordinoarcalis", "arinsal", "pal", "soldeu", "canillo", "encamp", "pas", "casa", "hotel", "hoteles",
    "office", "mail", "suport", "soporte", "support", "client", "clients", "cliente", "clientes",
    "equip", "equipo", "sistemes", "sistemas", "dades", "temporada", "informació", "informacion",
    "tots", "todos", "servei", "servicio", "servicios", "info", "central", "direcció", "direccion",
    "manager", "agent", "admin", "informàtica", "informatica", "comunicacions", "comunicaciones", "arcalís", "saetde", "emap",
}

_TAGS_LINE = re.compile(r"^Tags: .*$", re.M)


def surface_stats(texts):
    """Por palabra: total de apariciones, con inicial mayúscula y escritas enteras en MAYÚSCULAS."""
    total, cap, upper = Counter(), Counter(), Counter()
    for t in texts:
        for w in NAME_PART.findall(_TAGS_LINE.sub("", t)):
            total[w.lower()] += 1
            cap[w.lower()] += w[0].isupper()
            upper[w.lower()] += w.isupper()
    return total, cap, upper


def compute_exclude(users: dict, docs: list[dict]):
    """Palabras de nombres de usuario que NO deben tratarse como nombre de persona.

    Se calcula con el corpus de Zendesk (muestra grande); lo reutiliza también la ingesta de Notion.
    """
    candidates = {p for u in users.values() for p in name_parts(u.get("name"))}
    total, cap, upper = surface_stats(d["content"] for d in docs)
    tag_words = {w for d in docs for t in d["metadata"].get("tags") or [] for w in NAME_PART.findall(t.lower())}
    exclude, low_cap = set(ORG_WORDS), []
    for p in candidates:
        if not total[p]:
            continue
        if cap[p] / total[p] < MIN_CAPITALIZED:
            exclude.add(p)
            low_cap.append((total[p], p, cap[p] / total[p]))
        elif upper[p] / total[p] >= MAX_UPPER or p in tag_words:  # siglas y departamentos
            exclude.add(p)
            low_cap.append((total[p], p, -1))
    exclude |= {p for p in candidates if p in ORG_WORDS}
    return exclude, low_cap, candidates


EXCLUDE_CACHE = Path("data/exclude_parts.json")


def load_exclude(users: dict) -> set[str]:
    """Palabras excluidas de la seudonimización de nombres. Se calcula una vez (lee todo el corpus de
    Zendesk) y se guarda en data/exclude_parts.json; borrar ese fichero para recalcular."""
    if EXCLUDE_CACHE.exists():
        return set(json.loads(EXCLUDE_CACHE.read_text(encoding="utf-8")))
    docs = [json.loads(l) for l in PATH.open(encoding="utf-8")]
    exclude = compute_exclude(users, docs)[0]
    EXCLUDE_CACHE.write_text(json.dumps(sorted(exclude), ensure_ascii=False), encoding="utf-8")
    return exclude


def main():
    users = get_users_zendesk()
    docs = [json.loads(l) for l in PATH.open(encoding="utf-8")]

    exclude, low_cap, candidates = compute_exclude(users, docs)
    print(f"{len(candidates)} palabras candidatas; {len(exclude & candidates)} excluidas "
          f"({len(low_cap)} por no ir con mayúscula, el resto por ser organizaciones/palabras comunes)")

    pseudo = Pseudonymizer(users=users, exclude_parts=exclude)
    hits: Counter = Counter()

    def redact(text: str) -> str:
        # la línea de Tags se protege: son etiquetas, nunca nombres
        parts = re.split(r"(^Tags: .*$)", text, flags=re.M)
        for i in range(0, len(parts), 2):
            for w in NAME_PART.findall(parts[i]):
                if w.lower() in pseudo._parts:
                    hits[w.lower()] += 1
            parts[i] = pseudo.redact_people(Pseudonymizer.redact_secrets(parts[i]))
        return "".join(parts)

    changed = 0
    with TMP.open("w", encoding="utf-8") as dst:
        for d in docs:
            new_content, new_title = redact(d["content"]), redact(d["title"])
            changed += (new_content != d["content"]) or (new_title != d["title"])
            d["content"], d["title"] = new_content, new_title
            dst.write(json.dumps(d, ensure_ascii=False) + "\n")
    os.replace(TMP, PATH)
    print(f"{changed} de {len(docs)} documentos modificados")
    print("palabras sustituidas con más frecuencia (deben ser nombres de personas):")
    for w, n in hits.most_common(30):
        print(f"  {n:6d}  {w}  -> {pseudo._parts[w]}")
    print("excluidas por ir casi siempre en minúscula (muestra):")
    for n, p, r in sorted(low_cap, reverse=True)[:15]:
        print(f"  {n:6d}  {p}  (sigla/tag)" if r < 0 else f"  {n:6d}  {p}  ({r:.0%} con mayúscula)")


if __name__ == "__main__":
    main()
