from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import cv2

from service.engine import FaceEngine, resolve_stream_candidates
from service.storage import EventRecord, SNAPSHOTS_DIR, SourceRecord, Storage


ANALYZE_INTERVAL = 0.7
RECONNECT_DELAY = 2.0
EVENT_COOLDOWN = 8.0


class SourceWorker:
    def __init__(self, source: SourceRecord, storage: Storage, engine: FaceEngine) -> None:
        self.source = source
        self.storage = storage
        self.engine = engine
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.last_event_key = ""
        self.last_event_time = 0.0
        self.status = "stopped"
        self.current_url = ""
        self.last_error = ""
        self.last_person_name = ""
        self.last_score = 0.0
        self.last_detection_at = ""

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)

    def run(self) -> None:
        while not self.stop_event.is_set():
            capture = self.open_capture()
            if capture is None:
                self.status = "unavailable"
                self.last_error = "Unable to open source"
                time.sleep(RECONNECT_DELAY)
                continue

            self.status = "running"
            self.last_error = ""
            last_analysis = 0.0
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    self.status = "reconnecting"
                    self.last_error = "Frame read failed"
                    break
                now = time.time()
                if now - last_analysis < ANALYZE_INTERVAL:
                    time.sleep(0.03)
                    continue
                last_analysis = now
                people_cache = self.storage.list_people()
                try:
                    detections = self.engine.analyze_frame(frame, people_cache)
                except Exception as error:
                    self.last_error = str(error)
                    continue
                if detections:
                    self.handle_detection(detections[0], frame)
            capture.release()
            time.sleep(RECONNECT_DELAY)

    def open_capture(self) -> cv2.VideoCapture | None:
        local_capture = open_local_capture(self.source.url)
        if local_capture is not None:
            ok, frame = local_capture.read()
            if ok and frame is not None:
                return local_capture
            local_capture.release()

        for candidate in resolve_stream_candidates(self.source.url):
            capture = cv2.VideoCapture(candidate)
            if not capture.isOpened():
                capture.release()
                continue
            ok, frame = capture.read()
            if ok and frame is not None:
                self.current_url = candidate
                return capture
            capture.release()
        return None

    def handle_detection(self, detection: dict, frame) -> None:
        event_key = f"{self.source.source_id}:{detection['person_name']}:{detection['matched']}"
        now = time.time()
        if event_key == self.last_event_key and now - self.last_event_time < EVENT_COOLDOWN:
            return
        self.last_event_key = event_key
        self.last_event_time = now
        self.last_person_name = detection["person_name"]
        self.last_score = float(detection["score"])
        self.last_detection_at = datetime.now(timezone.utc).isoformat()
        snapshot_path = save_snapshot(self.source.source_id, int(now * 1000), frame, detection)
        event = EventRecord(
            event_id=f"evt_{int(now * 1000)}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            source_id=self.source.source_id,
            source_name=self.source.name,
            matched=bool(detection["matched"]),
            person_id=detection["person_id"],
            person_name=detection["person_name"],
            info=detection["info"],
            score=float(detection["score"]),
            snapshot_path=snapshot_path,
        )
        self.storage.append_event(event)

    def get_status(self) -> dict:
        return {
            "source_id": self.source.source_id,
            "source_name": self.source.name,
            "configured_url": self.source.url,
            "resolved_url": self.current_url,
            "status": self.status,
            "last_error": self.last_error,
            "last_person_name": self.last_person_name,
            "last_score": self.last_score,
            "last_detection_at": self.last_detection_at,
        }


class RuntimeManager:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        self.workers: dict[str, SourceWorker] = {}
        self.lock = threading.Lock()

    def start_enabled_sources(self) -> None:
        for source in self.storage.list_sources():
            if source.enabled:
                self.start_source(source.source_id)

    def stop_all(self) -> None:
        with self.lock:
            workers = list(self.workers.values())
            self.workers.clear()
        for worker in workers:
            worker.stop()

    def start_source(self, source_id: str) -> None:
        source = next((item for item in self.storage.list_sources() if item.source_id == source_id), None)
        if source is None:
            raise ValueError("Source not found")
        with self.lock:
            worker = self.workers.get(source_id)
            if worker is None:
                worker = SourceWorker(source, self.storage, FaceEngine())
                self.workers[source_id] = worker
            else:
                if worker.source.url != source.url or worker.source.enabled != source.enabled or worker.source.name != source.name:
                    old_worker = self.workers.pop(source_id)
                    old_worker.stop()
                    worker = SourceWorker(source, self.storage, FaceEngine())
                    self.workers[source_id] = worker
                else:
                    worker.source = source
        worker.start()

    def stop_source(self, source_id: str) -> None:
        with self.lock:
            worker = self.workers.pop(source_id, None)
        if worker is not None:
            worker.stop()

    def get_source_statuses(self) -> list[dict]:
        configured_sources = {source.source_id: source for source in self.storage.list_sources()}
        with self.lock:
            active_workers = dict(self.workers)

        statuses = []
        for source_id, source in configured_sources.items():
            worker = active_workers.get(source_id)
            if worker is not None:
                statuses.append(worker.get_status())
            else:
                statuses.append(
                    {
                        "source_id": source.source_id,
                        "source_name": source.name,
                        "configured_url": source.url,
                        "resolved_url": "",
                        "status": "disabled" if not source.enabled else "stopped",
                        "last_error": "",
                        "last_person_name": "",
                        "last_score": 0.0,
                        "last_detection_at": "",
                    }
                )
        return statuses


def open_local_capture(source_url: str) -> cv2.VideoCapture | None:
    normalized = source_url.strip()
    lowered = normalized.lower()

    if lowered.startswith("device://"):
        device_id = normalized.split("://", 1)[1]
        if device_id.isdigit():
            return cv2.VideoCapture(int(device_id))
        return cv2.VideoCapture(device_id)

    if lowered.startswith("v4l2://"):
        device_path = normalized.split("://", 1)[1]
        return cv2.VideoCapture(device_path)

    if lowered.startswith("/dev/video"):
        return cv2.VideoCapture(normalized)

    parsed = urlparse(normalized)
    if parsed.scheme == "file" and parsed.path.startswith("/dev/video"):
        return cv2.VideoCapture(parsed.path)

    return None


def save_snapshot(source_id: str, timestamp_ms: int, frame, detection: dict) -> str:
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_name = f"{source_id}_{timestamp_ms}.jpg"
    snapshot_path = SNAPSHOTS_DIR / snapshot_name

    image = frame.copy()
    x, y, w, h = detection.get("box", (0, 0, 0, 0))
    if w > 0 and h > 0:
        color = (0, 180, 0) if detection.get("matched") else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    cv2.imwrite(str(snapshot_path), image)
    return str(Path("service_data") / "snapshots" / snapshot_name)
