from zenpy import Zenpy, ZenpyException
import requests as requests
import os
from dotenv import load_dotenv
import zenpy

load_dotenv()

zenpy_client = Zenpy(
    email=os.getenv("ZENDESK_EMAIL"),
    token=os.getenv("ZENDESK_TOKEN"),
    subdomain=os.getenv("ZENDESK_SUBDOMAIN"),
)

DEFAULT_ASSIGNEE_NAMES = {n.strip() for n in os.getenv("ZENDESK_ASSIGNEES", "").split(",") if n.strip()}

#Me autentico con auth_token con la API de Zendesk si la autenticacion es correcta, me devuelve el usuario autenticado
def authenticate_with_zendesk():
    try:
        response = zenpy_client.users.me()
        print(response)
        return True
    except Exception as e:
        print(f"Error al autenticar con la API de Zendesk: {e}")
        exit(1)

def _get_client() -> Zenpy:
    return zenpy_client

############################ FUNCIONES AUXILIARES
def get_user_name(users_dict: dict, user_id: int) -> str | None:
    """Resuelve nombre legible por user_id, con fallback a email."""
    user = users_dict.get(user_id)
    if not user:
        return None
    return user.get("name") or user.get("email")

def _should_include_assignee(
    assignee_name: str | None,
    incluir_todos_assignees: bool,
    assignee_names: set[str] | None,
    ) -> bool:
    """Decide si un ticket entra por filtro de assignee."""
    if incluir_todos_assignees:
        return True

    allowed_names = assignee_names if assignee_names is not None else DEFAULT_ASSIGNEE_NAMES
    if not allowed_names:
        return True

    normalized_allowed = {_normalize_name(name) for name in allowed_names}
    return _normalize_name(assignee_name) in normalized_allowed

def _ticket_field(ticket, field_name: str):
    """Acceso uniforme a campos para dict (incremental) u objeto Zenpy (full)."""
    if isinstance(ticket, dict):
        return ticket.get(field_name)
    return getattr(ticket, field_name, None)

def get_custom_field_value_from_dict(ticket: dict, custom_field_id: int):
    """Lee custom field desde dict (modo incremental export API)."""
    custom_fields = ticket.get("custom_fields", []) or []
    for cf in custom_fields:
        if isinstance(cf, dict) and cf.get("id") == custom_field_id:
            return cf.get("value")
    return None

def get_custom_field_value(ticket, custom_field_id: int):
    """Lee custom field desde objeto Ticket de Zenpy (modo full_sync)."""
    # Zendesk custom_fields es una lista de objetos con id/value
    for cf in (getattr(ticket, "custom_fields", None) or []):
        if getattr(cf, "id", None) == custom_field_id:
            return getattr(cf, "value", None)
    return None

def get_users_zendesk():
    """Trae todos los usuarios de Zendesk una sola vez y arma un diccionario por id."""
    users_dict = {}
    for user in _get_client().users():
        users_dict[user.id] = {
            "name": user.name,
            "email": user.email,
        }
    return users_dict

def _normalize_name(value: str | None) -> str:
    return (value or "").strip().lower()

############################ FIN FUNCIONES AUXILIARES



#Consulto todos los tickets que hay en estado solved y closed
def get_tickets(

    incluir_todos_assignees: bool = False,
    assignee_names: set[str] | None = None,

    ):

    tickets = list(zenpy_client.tickets())


    users_dict = get_users_zendesk()
    custom_field_id_externo = 26458168061468
    custom_field_pendientede = 360012377511 

    tickets_dict = []
    for ticket in tickets:
        assignee_name = get_user_name(users_dict, _ticket_field(ticket, "assignee_id"))
        if not _should_include_assignee(assignee_name, incluir_todos_assignees, assignee_names):
            continue

        if isinstance(ticket, dict):
            id_externo = get_custom_field_value_from_dict(ticket, custom_field_id_externo)
            pendiente_de =  get_custom_field_value_from_dict(ticket, custom_field_pendientede)
        else:
            id_externo = get_custom_field_value(ticket, custom_field_id_externo)
            pendiente_de = get_custom_field_value(ticket, custom_field_pendientede)

        tickets_dict.append({
            "id": _ticket_field(ticket, "id"),
            "subject": _ticket_field(ticket, "subject"),
            "status": _ticket_field(ticket, "status"),
            # "requester_name": get_user_name(users_dict, _ticket_field(ticket, "requester_id")),
            "assignee_name": get_user_name(users_dict, _ticket_field(ticket, "assignee_id")),
            "created_at": _ticket_field(ticket, "created_at"),
            "updated_at": _ticket_field(ticket, "updated_at"),
            "url": _ticket_field(ticket, "url"),
            'pendiente_de': pendiente_de,
            "id_externo": id_externo

        })
            


    return tickets_dict

CLOSED_STATUSES = {"solved", "closed"}
custom_field_id_externo = 26458168061468
custom_field_pendientede = 360012377511

def iter_closed_tickets(limit: int | None = None):
    """Genera tickets solved/closed con tags, descripción y comentarios.

    ATENCIÓN: devuelve texto SIN tratar, solo para procesarlo en memoria.
    No escribir su salida a disco sin pasar antes por Pseudonymizer.
    """
    client = _get_client()
    users_dict = get_users_zendesk()
    yield users_dict  # el primer elemento son los usuarios, para construir el Pseudonymizer

    # OJO: client.tickets() NO devuelve los tickets archivados (cerrados antiguos).
    # La exportación incremental sí los incluye, por eso se usa aquí.
    count = 0
    seen = set()
    for ticket in client.tickets.incremental(start_time=0):
        if _ticket_field(ticket, "status") not in CLOSED_STATUSES or ticket.id in seen:
            continue
        seen.add(ticket.id)

        comments = []
        for c in client.tickets.comments(ticket=ticket.id):
            comments.append({
                "author_id": c.author_id,
                "public": bool(getattr(c, "public", True)),
                "created_at": c.created_at,
                "body": getattr(c, "plain_body", None) or c.body or "",
            })

        yield {
            "id": ticket.id,
            "subject": ticket.subject,
            "status": ticket.status,
            "description": ticket.description,
            "tags": list(ticket.tags or []),
            "requester_id": ticket.requester_id,
            "assignee_name": get_user_name(users_dict, ticket.assignee_id),
            "created_at": ticket.created_at,
            "updated_at": ticket.updated_at,
            "url": ticket.url,
            "pendiente_de": get_custom_field_value(ticket, custom_field_pendientede),
            "id_externo": get_custom_field_value(ticket, custom_field_id_externo),
            "comments": comments,
        }
        count += 1
        if limit and count >= limit:
            return

if __name__ == "__main__":
    zdk = get_tickets()
    print(zdk)
