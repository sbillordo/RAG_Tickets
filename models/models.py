from dataclasses import dataclass, asdict, field
from datetime import datetime

import json


@dataclass
class Document:
    id: str
    source: str
    status: str
    title: str
    content: str
    created_at: datetime
    updated_at: datetime
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            raise ValueError("id no puede estar vacío")

    def to_dict(self):
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        d["updated_at"] = self.updated_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict):
        d = dict(d)  # copia para no mutar el original
        d["created_at"] = datetime.fromisoformat(d["created_at"])
        d["updated_at"] = datetime.fromisoformat(d["updated_at"])
        return cls(**d)


if __name__ == "__main__":
    doc = Document(
        id="1", source="zendesk", status="open",
        title="prueba", content="contenido de prueba",
        created_at=datetime.now(), updated_at=datetime.now(),
    )
    print(json.dumps(doc.to_dict()))
    linea = json.dumps(doc.to_dict())
    doc2 = Document.from_dict(json.loads(linea))
    print(doc2)
    print(doc == doc2)