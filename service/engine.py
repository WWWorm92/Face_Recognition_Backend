import re
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import cv2
import numpy as np

from service.storage import MODELS_DIR, PersonRecord, RecognitionSettings, SourceRecord


YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
YUNET_MODEL_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_MODEL_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
FACE_ANGLES = (0, -20, 20, -10, 10)
PROCESS_WIDTH = 480


def ensure_model(model_path: Path, model_url: str) -> None:
    if model_path.exists() and is_valid_model_file(model_path):
        return
    with urllib.request.urlopen(model_url, timeout=30) as response:
        model_path.write_bytes(response.read())
    if not is_valid_model_file(model_path):
        raise RuntimeError(f"Invalid model file: {model_path.name}")


def is_valid_model_file(model_path: Path) -> bool:
    if not model_path.exists() or model_path.stat().st_size < 100_000:
        return False
    header = model_path.read_bytes()[:64]
    return not header.startswith(b"version https://git-lfs.github.com/spec/v1") and not header.lstrip().startswith(b"<html")


def decode_image(image_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Не удалось прочитать изображение")
    return image


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def resize_for_processing(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    if width <= PROCESS_WIDTH:
        return image, 1.0
    scale = PROCESS_WIDTH / width
    return cv2.resize(image, (int(width * scale), int(height * scale))), scale


def as_embedding_array(embedding: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(embedding, dtype=np.float32).reshape(1, -1)


def generate_aligned_variants(aligned_face: np.ndarray) -> list[np.ndarray]:
    variants = [
        aligned_face,
        cv2.flip(aligned_face, 1),
        cv2.convertScaleAbs(aligned_face, alpha=1.05, beta=10),
        cv2.convertScaleAbs(aligned_face, alpha=0.95, beta=-10),
        cv2.GaussianBlur(aligned_face, (3, 3), 0),
    ]
    rotated = []
    for variant in variants:
        for angle in FACE_ANGLES:
            rotated.append(rotate_image(variant, angle))
    return variants + rotated


def fetch_remote_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=15) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="ignore")


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def build_flussonic_candidates_from_url(source_url: str) -> list[str]:
    parsed = urlparse(source_url)
    query = parse_qs(parsed.query)
    token = query.get("token", [""])[0]
    path = parsed.path.rstrip("/")
    if path.endswith("/whep"):
        path = path[: -len("/whep")]
    candidates = []
    for suffix in ("/index.m3u8", "/mpegts", "/archive-0/index.m3u8"):
        candidate_query = urlencode({"token": token}) if token else ""
        candidates.append(urlunparse((parsed.scheme, parsed.netloc, f"{path}{suffix}", "", candidate_query, "")))
    return candidates


def extract_stream_candidates_from_html(source_url: str) -> list[str]:
    html_text = fetch_remote_text(source_url)
    direct_links = re.findall(r"https?://[^\s\"'<>]+", html_text)
    candidates: list[str] = []
    for link in direct_links:
        lower = link.lower()
        if ".m3u8" in lower or lower.startswith("rtsp://") or "/mpegts" in lower:
            candidates.append(link)
        elif "/whep" in lower:
            candidates.extend(build_flussonic_candidates_from_url(link))
    parsed = urlparse(source_url)
    query = parse_qs(parsed.query)
    video = query.get("video", [""])[0]
    token = query.get("token", [""])[0]
    if parsed.netloc.endswith("etd-online.ru") and video and token:
        candidates.append(f"https://flussonic.etd-site.ru/{video}/index.m3u8?token={token}")
        candidates.append(f"https://flussonic.etd-site.ru/{video}/mpegts?token={token}")
    return unique_preserve_order(candidates)


def resolve_stream_candidates(source_url: str) -> list[str]:
    lower = source_url.lower().strip()
    if lower.startswith("rtsp://") or ".m3u8" in lower or "/mpegts" in lower or ".mjpg" in lower or ".mp4" in lower:
        return [source_url]
    if "/whep" in lower:
        return build_flussonic_candidates_from_url(source_url)
    if lower.endswith(".html") or ".html?" in lower:
        return extract_stream_candidates_from_html(source_url)
    return [source_url]


class FaceEngine:
    def __init__(self, settings: RecognitionSettings | None = None) -> None:
        ensure_model(YUNET_MODEL_PATH, YUNET_MODEL_URL)
        ensure_model(SFACE_MODEL_PATH, SFACE_MODEL_URL)
        self.detector = cv2.FaceDetectorYN.create(str(YUNET_MODEL_PATH), "", (640, 480), 0.75, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL_PATH), "")
        self.settings = settings or RecognitionSettings()

    def set_settings(self, settings: RecognitionSettings) -> None:
        self.settings = settings

    def detect_faces(self, image: np.ndarray) -> list[np.ndarray]:
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None:
            return []
        filtered_faces = []
        for face in faces:
            w = float(face[2])
            h = float(face[3])
            score = float(face[-1])
            if score < self.settings.detection_score_threshold:
                continue
            if w < self.settings.min_face_width or h < self.settings.min_face_height:
                continue
            if (w * h) < self.settings.min_face_area:
                continue
            filtered_faces.append(face)
        return sorted(filtered_faces, key=lambda face: float(face[2] * face[3] * face[-1]), reverse=True)

    def detect_best_face(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        best_image = None
        best_face = None
        best_score = -1.0
        for angle in FACE_ANGLES:
            rotated = rotate_image(image, angle) if angle else image
            faces = self.detect_faces(rotated)
            if not faces:
                continue
            face = faces[0]
            score = float(face[2] * face[3] * face[-1])
            if score > best_score:
                best_score = score
                best_image = rotated
                best_face = face
        if best_image is None or best_face is None:
            raise ValueError("На фото не найдено лицо")
        return best_image, best_face

    def extract_embeddings_from_bytes(self, image_bytes: bytes) -> list[list[float]]:
        image = decode_image(image_bytes)
        detected_image, face = self.detect_best_face(image)
        aligned_face = self.recognizer.alignCrop(detected_image, face)
        if aligned_face is None or aligned_face.size == 0:
            raise ValueError("Не удалось выровнять лицо")
        return [self.recognizer.feature(variant).flatten().astype(float).tolist() for variant in generate_aligned_variants(aligned_face)]

    def match_embedding(self, query_embedding: np.ndarray, records: list[PersonRecord]) -> tuple[PersonRecord | None, float]:
        query_embedding = as_embedding_array(query_embedding)
        best_record = None
        best_score = -1.0
        for record in records:
            for stored_embedding in record.embeddings:
                stored = as_embedding_array(stored_embedding)
                if stored.shape != query_embedding.shape:
                    continue
                score = float(self.recognizer.match(query_embedding, stored, cv2.FaceRecognizerSF_FR_COSINE))
                if score > best_score:
                    best_score = score
                    best_record = record
        return best_record, best_score

    def analyze_frame(self, image: np.ndarray, records: list[PersonRecord], source: SourceRecord | None = None) -> list[dict]:
        region_image = image
        offset_x = 0
        offset_y = 0
        if source and source.roi_enabled:
            height, width = image.shape[:2]
            x1 = max(0, min(width - 1, int(width * source.roi_x)))
            y1 = max(0, min(height - 1, int(height * source.roi_y)))
            x2 = max(x1 + 1, min(width, int(width * (source.roi_x + source.roi_w))))
            y2 = max(y1 + 1, min(height, int(height * (source.roi_y + source.roi_h))))
            region_image = image[y1:y2, x1:x2]
            offset_x = x1
            offset_y = y1

        scaled_image, scale = resize_for_processing(region_image)
        faces = self.detect_faces(scaled_image)
        result = []
        for face in faces:
            scaled_face = face.copy()
            scaled_face[:14] = scaled_face[:14] / scale
            scaled_face[0] += offset_x
            scaled_face[1] += offset_y
            scaled_face[4:14:2] += offset_x
            scaled_face[5:14:2] += offset_y
            x, y, w, h = [int(value) for value in scaled_face[:4]]
            aligned_face = self.recognizer.alignCrop(image, scaled_face)
            if aligned_face is None or aligned_face.size == 0:
                continue
            query_embedding = self.recognizer.feature(aligned_face)
            record, score = self.match_embedding(query_embedding, records)
            matched = record is not None and score >= self.settings.cosine_threshold
            result.append(
                {
                    "box": (x, y, w, h),
                    "matched": matched,
                    "person_id": record.person_id if matched and record else None,
                    "person_name": record.name if matched and record else "Unknown",
                    "info": record.info if matched and record else "Нет данных",
                    "score": round(float(score), 3),
                }
            )
        return result
