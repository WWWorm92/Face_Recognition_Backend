import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "service_data"
PEOPLE_DIR = DATA_DIR / "people"
MODELS_DIR = DATA_DIR / "models"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
PEOPLE_DB = DATA_DIR / "people.json"
SOURCES_DB = DATA_DIR / "sources.json"
EVENTS_DB = DATA_DIR / "events.json"


@dataclass
class PersonRecord:
    person_id: str
    name: str
    info: str
    image_paths: list[str] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)


@dataclass
class SourceRecord:
    source_id: str
    name: str
    url: str
    enabled: bool = True


@dataclass
class EventRecord:
    event_id: str
    timestamp: str
    source_id: str
    source_name: str
    matched: bool
    person_id: str | None
    person_name: str
    info: str
    score: float
    snapshot_path: str = ""


class Storage:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        PEOPLE_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        for path in (PEOPLE_DB, SOURCES_DB, EVENTS_DB):
            if not path.exists():
                path.write_text("[]", encoding="utf-8")

    def _load_json(self, path: Path) -> list[dict[str, Any]]:
        self.ensure_dirs()
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_json(self, path: Path, payload: list[dict[str, Any]]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_people(self) -> list[PersonRecord]:
        with self.lock:
            return [PersonRecord(**item) for item in self._load_json(PEOPLE_DB)]

    def save_people(self, people: list[PersonRecord]) -> None:
        with self.lock:
            self._save_json(PEOPLE_DB, [asdict(item) for item in people])

    def list_sources(self) -> list[SourceRecord]:
        with self.lock:
            return [SourceRecord(**item) for item in self._load_json(SOURCES_DB)]

    def save_sources(self, sources: list[SourceRecord]) -> None:
        with self.lock:
            self._save_json(SOURCES_DB, [asdict(item) for item in sources])

    def list_events(self) -> list[EventRecord]:
        with self.lock:
            return [EventRecord(**item) for item in self._load_json(EVENTS_DB)]

    def append_event(self, event: EventRecord, limit: int = 200) -> None:
        with self.lock:
            events = [EventRecord(**item) for item in self._load_json(EVENTS_DB)]
            events.append(event)
            events = events[-limit:]
            self._save_json(EVENTS_DB, [asdict(item) for item in events])

    def update_person(self, person: PersonRecord) -> None:
        people = self.list_people()
        updated = [person if item.person_id == person.person_id else item for item in people]
        self.save_people(updated)

    def delete_person(self, person_id: str) -> PersonRecord | None:
        people = self.list_people()
        removed = None
        keep = []
        for item in people:
            if item.person_id == person_id:
                removed = item
            else:
                keep.append(item)
        if removed is not None:
            self.save_people(keep)
        return removed

    def add_source(self, name: str, url: str, enabled: bool) -> SourceRecord:
        sources = self.list_sources()
        source = SourceRecord(source_id=uuid.uuid4().hex, name=name.strip(), url=url.strip(), enabled=enabled)
        sources.append(source)
        self.save_sources(sources)
        return source

    def update_source(self, source: SourceRecord) -> None:
        sources = self.list_sources()
        updated = [source if item.source_id == source.source_id else item for item in sources]
        self.save_sources(updated)

    def delete_source(self, source_id: str) -> SourceRecord | None:
        sources = self.list_sources()
        removed = None
        keep = []
        for item in sources:
            if item.source_id == source_id:
                removed = item
            else:
                keep.append(item)
        if removed is not None:
            self.save_sources(keep)
        return removed
