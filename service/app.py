import os
from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from service.engine import FaceEngine
from service.runtime import RuntimeManager
from service.schemas import SourceCreate, SourceUpdate
from service.storage import BASE_DIR, DATA_DIR, PEOPLE_DIR, PersonRecord, RecognitionSettings, Storage


app = FastAPI(title="Home Assistant Face Recognition", version="0.1.0")
storage = Storage()
runtime = RuntimeManager(storage)
API_TOKEN = os.getenv("FACE_API_TOKEN", "").strip()
app.mount("/service_data", StaticFiles(directory=str(DATA_DIR)), name="service_data")


def require_auth(authorization: str | None = Header(default=None)) -> None:
    if not API_TOKEN:
        return
    expected = f"Bearer {API_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def token_is_valid(token: str | None, authorization: str | None = None) -> bool:
    if not API_TOKEN:
        return True
    if token == API_TOKEN:
        return True
    return authorization == f"Bearer {API_TOKEN}"


def ensure_ui_token(token: str | None) -> None:
    if API_TOKEN and token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


def redirect_with_token(token: str | None) -> RedirectResponse:
    suffix = f"?token={token}" if token else ""
    return RedirectResponse(url=f"/{suffix}", status_code=303)


def html_page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        f"""
        <!doctype html>
        <html lang=\"ru\">
        <head>
          <meta charset=\"utf-8\">
          <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
          <title>{title}</title>
          <style>
            body {{ font-family: Inter, Arial, sans-serif; background:#0b1220; color:#f8fafc; margin:0; padding:24px; }}
            h1,h2,h3 {{ margin:0 0 12px; }}
            .grid {{ display:grid; grid-template-columns:1.12fr 1fr; gap:20px; align-items:start; }}
            .row {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
            .card {{ background:linear-gradient(180deg,#172033,#101827); border:1px solid #263247; border-radius:16px; padding:18px; margin-bottom:18px; box-shadow:0 10px 30px rgba(0,0,0,0.18); }}
            .item {{ border-top:1px solid #2b3649; padding-top:14px; margin-top:14px; }}
            .muted {{ color:#9fb0c8; font-size:14px; }}
            .metric {{ font-size:28px; font-weight:700; }}
            .mono {{ font-family:Consolas, monospace; word-break:break-all; }}
            .snapshot {{ margin-top:10px; max-width:220px; border-radius:10px; border:1px solid #374151; display:block; }}
            .thumb {{ width:92px; height:92px; object-fit:cover; border-radius:10px; border:1px solid #374151; display:block; }}
            .thumb-grid {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }}
            .pill {{ display:inline-block; padding:4px 10px; border-radius:999px; background:#0b1220; border:1px solid #314056; font-size:12px; }}
            .section-title {{ display:flex; justify-content:space-between; align-items:center; gap:10px; }}
            input, textarea {{ width:100%; padding:10px; border-radius:10px; border:1px solid #374151; background:#0b1220; color:#f8fafc; box-sizing:border-box; }}
            textarea {{ min-height:88px; resize:vertical; }}
            button {{ padding:10px 14px; border:none; border-radius:10px; background:#2563eb; color:white; cursor:pointer; }}
            button.secondary {{ background:#334155; }}
            button.danger {{ background:#b91c1c; }}
            a {{ color:#93c5fd; text-decoration:none; }}
            label {{ font-size:13px; color:#cbd5e1; display:block; margin-bottom:4px; }}
          </style>
        </head>
        <body>{body}</body>
        </html>
        """
    )


def html_error_page(message: str, token: str | None = None) -> HTMLResponse:
    retry_link = f"/?token={token}" if token else "/"
    return html_page(
        "Face Recognition Admin Error",
        f"""
        <div class='card' style='max-width:720px;margin:60px auto;'>
          <h1>Ошибка</h1>
          <p>{message}</p>
          <div class='row' style='margin-top:16px;'>
            <a href='{retry_link}'>Вернуться в панель</a>
          </div>
        </div>
        """,
    )


def save_person_images(person_id: str, files: list[UploadFile], start_index: int = 1) -> list[str]:
    image_paths = []
    for offset, upload in enumerate(files, start=start_index):
        image_bytes = upload.file.read()
        extension = Path(upload.filename or "photo.jpg").suffix.lower() or ".jpg"
        image_path = PEOPLE_DIR / f"{person_id}_{offset}{extension}"
        image_path.write_bytes(image_bytes)
        image_paths.append(str(image_path.relative_to(BASE_DIR)))
    return image_paths


def clamp_roi(value: float) -> float:
    return max(0.0, min(1.0, value))


def rebuild_person_embeddings(person: PersonRecord) -> PersonRecord:
    engine = FaceEngine(storage.get_settings())
    embeddings = []
    valid_paths = []
    for path in person.image_paths:
        file_path = BASE_DIR / path
        if not file_path.exists():
            continue
        embeddings.extend(engine.extract_embeddings_from_bytes(file_path.read_bytes()))
        valid_paths.append(path)
    person.image_paths = valid_paths
    person.embeddings = embeddings
    return person


@app.on_event("startup")
def startup() -> None:
    runtime.start_enabled_sources()


@app.on_event("shutdown")
def shutdown() -> None:
    runtime.stop_all()


@app.get("/", response_class=HTMLResponse)
def admin_page(token: str | None = None, authorization: str | None = Header(default=None)) -> HTMLResponse:
    if not token_is_valid(token, authorization):
        return html_page(
            "Face Recognition Admin",
            """
            <div class='card' style='max-width:420px;margin:60px auto;'>
              <h1>Face Recognition Admin</h1>
              <p class='muted'>Введите API token.</p>
              <form method='get' action='/'>
                <input type='password' name='token' placeholder='API token'>
                <div style='height:12px'></div>
                <button type='submit'>Войти</button>
              </form>
            </div>
            """,
        )

    people = storage.list_people()
    sources = storage.list_sources()
    settings = storage.get_settings()
    statuses = {item["source_id"]: item for item in runtime.get_source_statuses()}
    events = storage.list_events()[-20:][::-1]

    people_html = "".join(
        f"""
        <div class='item'>
          <div class='section-title'><h3>{person.name}</h3><span class='pill'>{len(person.image_paths)} фото</span></div>
          <div class='muted'>{person.info or 'Без дополнительных данных'}</div>
          <div class='row' style='margin-top:10px;'>
            <span class='pill'>Эмбеддинги: <b>{len(person.embeddings)}</b></span>
          </div>
          <div class='thumb-grid'>
            {''.join(f"<div><a href='/{path}' target='_blank'><img class='thumb' src='/{path}' alt='photo'></a><form method='post' action='/ui/people/{person.person_id}/photos/delete?token={token or ''}' style='margin-top:6px;'><input type='hidden' name='photo_path' value='{path}'><button class='secondary' type='submit'>Удалить фото</button></form></div>" for path in person.image_paths)}
          </div>
          <div class='row' style='margin-top:12px;'>
            <form method='post' action='/ui/people/{person.person_id}/photos?token={token or ""}' enctype='multipart/form-data'>
              <input type='file' name='photos' multiple required>
              <div style='height:8px'></div>
              <button type='submit'>Добавить фото</button>
            </form>
            <form method='post' action='/ui/people/{person.person_id}/delete?token={token or ""}'>
              <button class='danger' type='submit'>Удалить человека</button>
            </form>
          </div>
        </div>
        """
        for person in people
    ) or "<div class='muted'>Людей пока нет.</div>"

    sources_html = "".join(
        f"""
        <div class='item'>
          <div class='section-title'><h3>{source.name}</h3><span class='pill'>{statuses.get(source.source_id, {}).get('status', 'unknown')}</span></div>
          <div class='mono muted'>{source.url}</div>
          <div class='row' style='margin-top:10px;'>
            <span class='pill'>Последний человек: <b>{statuses.get(source.source_id, {}).get('last_person_name', '') or '-'}</b></span>
            <span class='pill'>Score: <b>{statuses.get(source.source_id, {}).get('last_score', 0.0)}</b></span>
          </div>
          <div class='muted'>ROI: {'on' if source.roi_enabled else 'off'} ({source.roi_x:.2f}, {source.roi_y:.2f}, {source.roi_w:.2f}, {source.roi_h:.2f})</div>
          <div class='muted'>Resolved URL: {statuses.get(source.source_id, {}).get('resolved_url', '') or '-'}</div>
          <div class='muted'>Ошибка: {statuses.get(source.source_id, {}).get('last_error', '') or '-'}</div>
          <div style='margin-top:12px;'>
            <div class='muted'>Нарисуй ROI мышью на кадре и потом сохрани.</div>
            <div id='roi-wrap-{source.source_id}' style='position:relative;display:inline-block;margin-top:8px;border:1px solid #314056;border-radius:12px;overflow:hidden;'>
              <img id='roi-image-{source.source_id}' src='/ui/sources/{source.source_id}/preview?token={token or ""}' alt='roi preview' style='display:block;max-width:100%;width:360px;background:#020617;'>
              <div id='roi-box-{source.source_id}' style='position:absolute;border:2px solid #22c55e;background:rgba(34,197,94,0.16);left:{source.roi_x * 100}%;top:{source.roi_y * 100}%;width:{source.roi_w * 100}%;height:{source.roi_h * 100}%;pointer-events:none;'></div>
            </div>
          </div>
          <form method='post' action='/ui/sources/{source.source_id}/roi?token={token or ""}' style='margin-top:12px;'>
            <div class='row'>
              <label><input type='checkbox' name='roi_enabled' {'checked' if source.roi_enabled else ''}> Использовать ROI</label>
            </div>
            <div class='row' style='margin-top:8px;'>
              <div style='flex:1'><label>X</label><input id='roi-x-{source.source_id}' name='roi_x' type='number' min='0' max='1' step='0.01' value='{source.roi_x}'></div>
              <div style='flex:1'><label>Y</label><input id='roi-y-{source.source_id}' name='roi_y' type='number' min='0' max='1' step='0.01' value='{source.roi_y}'></div>
              <div style='flex:1'><label>W</label><input id='roi-w-{source.source_id}' name='roi_w' type='number' min='0.01' max='1' step='0.01' value='{source.roi_w}'></div>
              <div style='flex:1'><label>H</label><input id='roi-h-{source.source_id}' name='roi_h' type='number' min='0.01' max='1' step='0.01' value='{source.roi_h}'></div>
            </div>
            <div style='height:8px'></div>
            <div class='row'>
              <button class='secondary' type='button' onclick="reloadPreview('{source.source_id}', '{token or ''}')">Обновить кадр</button>
              <button class='secondary' type='submit'>Сохранить ROI</button>
            </div>
          </form>
          <div class='row' style='margin-top:12px;'>
            <form method='post' action='/ui/sources/{source.source_id}/start?token={token or ""}'><button type='submit'>Старт</button></form>
            <form method='post' action='/ui/sources/{source.source_id}/stop?token={token or ""}'><button class='secondary' type='submit'>Стоп</button></form>
            <form method='post' action='/ui/sources/{source.source_id}/delete?token={token or ""}'><button class='danger' type='submit'>Удалить</button></form>
          </div>
        </div>
        """
        for source in sources
    ) or "<div class='muted'>Источников пока нет.</div>"

    events_html = "".join(
        f"<div class='item'><b>{event.source_name}</b> | {event.person_name} | score {event.score} | {event.timestamp}"
        + (f" | <a href='/{event.snapshot_path}' target='_blank'>snapshot</a>" if event.snapshot_path else "")
        + (f"<br><a href='/{event.snapshot_path}' target='_blank'><img class='snapshot' src='/{event.snapshot_path}' alt='snapshot'></a>" if event.snapshot_path else "")
        + "</div>"
        for event in events
    ) or "<div class='muted'>Событий пока нет.</div>"

    body = f"""
    <h1>Face Recognition Admin</h1>
    <div class='row'>
      <div class='card'><div class='muted'>Людей</div><div class='metric'>{len(people)}</div></div>
      <div class='card'><div class='muted'>Источников</div><div class='metric'>{len(sources)}</div></div>
      <div class='card'><div class='muted'>Событий</div><div class='metric'>{len(storage.list_events())}</div></div>
      <div class='card'><div class='muted'>API Docs</div><div><a href='/docs'>/docs</a></div></div>
    </div>
    <div class='grid'>
      <div>
        <div class='card'>
          <h2>Добавить человека</h2>
          <form method='post' action='/ui/people?token={token or ""}' enctype='multipart/form-data'>
            <input name='name' placeholder='Имя' required>
            <div style='height:10px'></div>
            <textarea name='info' placeholder='Данные'></textarea>
            <div style='height:10px'></div>
            <input type='file' name='photos' multiple required>
            <div style='height:10px'></div>
            <button type='submit'>Сохранить</button>
          </form>
        </div>
        <div class='card'>
          <h2>Люди</h2>
          {people_html}
        </div>
      </div>
      <div>
        <div class='card'>
          <h2>Настройки точности</h2>
          <form method='post' action='/ui/settings?token={token or ""}'>
            <div class='row'>
              <div style='flex:1'><label>Порог совпадения</label><input name='cosine_threshold' type='number' min='0' max='1' step='0.01' value='{settings.cosine_threshold}' required></div>
              <div style='flex:1'><label>Порог детекции</label><input name='detection_score_threshold' type='number' min='0' max='1' step='0.01' value='{settings.detection_score_threshold}' required></div>
            </div>
            <div style='height:10px'></div>
            <div class='row'>
              <div style='flex:1'><label>Мин. ширина лица</label><input name='min_face_width' type='number' min='1' step='1' value='{settings.min_face_width}' required></div>
              <div style='flex:1'><label>Мин. высота лица</label><input name='min_face_height' type='number' min='1' step='1' value='{settings.min_face_height}' required></div>
            </div>
            <div style='height:10px'></div>
            <div class='row'>
              <div style='flex:1'><label>Мин. площадь лица</label><input name='min_face_area' type='number' min='1' step='1' value='{settings.min_face_area}' required></div>
              <div style='flex:1'><label>Подтверждений подряд</label><input name='confirmation_frames' type='number' min='1' step='1' value='{settings.confirmation_frames}' required></div>
            </div>
            <div style='height:10px'></div>
            <button type='submit'>Сохранить настройки</button>
          </form>
        </div>
        <div class='card'>
          <h2>Добавить источник</h2>
          <form method='post' action='/ui/sources?token={token or ""}'>
            <input name='name' placeholder='Имя источника' required>
            <div style='height:10px'></div>
            <input name='url' placeholder='rtsp://..., device://0, whep/html URL' required>
            <div style='height:10px'></div>
            <label><input type='checkbox' name='enabled' checked> Включить сразу</label>
            <div style='height:10px'></div>
            <button type='submit'>Добавить источник</button>
          </form>
        </div>
        <div class='card'>
          <h2>Источники</h2>
          {sources_html}
        </div>
        <div class='card'>
          <h2>Последние события</h2>
          {events_html}
        </div>
      </div>
    </div>
    <script>
      function reloadPreview(sourceId, token) {{
        const img = document.getElementById(`roi-image-${{sourceId}}`);
        if (!img) return;
        const suffix = token ? `?token=${{encodeURIComponent(token)}}&t=${{Date.now()}}` : `?t=${{Date.now()}}`;
        img.src = `/ui/sources/${{sourceId}}/preview` + suffix;
      }}

      function setupRoiEditor(sourceId) {{
        const wrap = document.getElementById(`roi-wrap-${{sourceId}}`);
        const box = document.getElementById(`roi-box-${{sourceId}}`);
        const inputX = document.getElementById(`roi-x-${{sourceId}}`);
        const inputY = document.getElementById(`roi-y-${{sourceId}}`);
        const inputW = document.getElementById(`roi-w-${{sourceId}}`);
        const inputH = document.getElementById(`roi-h-${{sourceId}}`);
        if (!wrap || !box || !inputX || !inputY || !inputW || !inputH) return;

        let startX = 0;
        let startY = 0;
        let drawing = false;

        function applyBox(x, y, w, h) {{
          box.style.left = `${{x * 100}}%`;
          box.style.top = `${{y * 100}}%`;
          box.style.width = `${{w * 100}}%`;
          box.style.height = `${{h * 100}}%`;
          inputX.value = x.toFixed(2);
          inputY.value = y.toFixed(2);
          inputW.value = Math.max(0.01, w).toFixed(2);
          inputH.value = Math.max(0.01, h).toFixed(2);
        }}

        wrap.addEventListener('mousedown', (event) => {{
          const rect = wrap.getBoundingClientRect();
          startX = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
          startY = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
          drawing = true;
          applyBox(startX, startY, 0.01, 0.01);
          event.preventDefault();
        }});

        window.addEventListener('mousemove', (event) => {{
          if (!drawing) return;
          const rect = wrap.getBoundingClientRect();
          const currentX = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
          const currentY = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
          const x = Math.min(startX, currentX);
          const y = Math.min(startY, currentY);
          const w = Math.abs(currentX - startX);
          const h = Math.abs(currentY - startY);
          applyBox(x, y, Math.max(0.01, w), Math.max(0.01, h));
        }});

        window.addEventListener('mouseup', () => {{
          drawing = false;
        }});
      }}

      {''.join(f"setupRoiEditor('{source.source_id}');" for source in sources)}
    </script>
    """
    return html_page("Face Recognition Admin", body)


@app.get('/ui/sources/{source_id}/preview')
def source_preview_ui(source_id: str, token: str | None = None):
    ensure_ui_token(token)
    try:
        frame = runtime.capture_preview(source_id)
    except Exception as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    ok, encoded = cv2.imencode('.jpg', frame)
    if not ok:
        raise HTTPException(status_code=500, detail='Failed to encode preview')
    return Response(content=encoded.tobytes(), media_type='image/jpeg')


@app.post("/ui/people")
def create_person_ui(token: str | None = None, name: str = Form(...), info: str = Form(""), photos: list[UploadFile] = File(...)):
    ensure_ui_token(token)
    try:
        create_person(name=name, info=info, photos=photos)
    except HTTPException as error:
        return html_error_page(str(error.detail), token)
    except Exception as error:
        return html_error_page(str(error), token)
    return redirect_with_token(token)


@app.post("/ui/people/{person_id}/delete")
def delete_person_ui(person_id: str, token: str | None = None):
    ensure_ui_token(token)
    delete_person(person_id)
    return redirect_with_token(token)


@app.post("/ui/people/{person_id}/photos")
def append_person_photos_ui(person_id: str, token: str | None = None, photos: list[UploadFile] = File(...)):
    ensure_ui_token(token)
    try:
        append_person_photos(person_id, photos)
    except HTTPException as error:
        return html_error_page(str(error.detail), token)
    except Exception as error:
        return html_error_page(str(error), token)
    return redirect_with_token(token)


@app.post("/ui/people/{person_id}/photos/delete")
def delete_person_photo_ui(person_id: str, token: str | None = None, photo_path: str = Form(...)):
    ensure_ui_token(token)
    person = next((item for item in storage.list_people() if item.person_id == person_id), None)
    if person is None:
        return html_error_page("Person not found", token)
    if photo_path not in person.image_paths:
        return html_error_page("Photo not found in person profile", token)
    if len(person.image_paths) <= 1:
        return html_error_page("Нельзя удалить последнее фото человека. Сначала добавь новое фото или удали человека целиком.", token)
    absolute_path = BASE_DIR / photo_path
    if absolute_path.exists():
        absolute_path.unlink()
    person.image_paths = [path for path in person.image_paths if path != photo_path]
    try:
        person = rebuild_person_embeddings(person)
    except Exception as error:
        return html_error_page(str(error), token)
    storage.update_person(person)
    return redirect_with_token(token)


@app.post("/ui/sources")
def create_source_ui(token: str | None = None, name: str = Form(...), url: str = Form(...), enabled: str | None = Form(None)):
    ensure_ui_token(token)
    try:
        create_source(SourceCreate(name=name, url=url, enabled=enabled is not None))
    except HTTPException as error:
        return html_error_page(str(error.detail), token)
    except Exception as error:
        return html_error_page(str(error), token)
    return redirect_with_token(token)


@app.post("/ui/sources/{source_id}/roi")
def update_source_roi_ui(
    source_id: str,
    token: str | None = None,
    roi_enabled: str | None = Form(None),
    roi_x: float = Form(...),
    roi_y: float = Form(...),
    roi_w: float = Form(...),
    roi_h: float = Form(...),
):
    ensure_ui_token(token)
    source = next((item for item in storage.list_sources() if item.source_id == source_id), None)
    if source is None:
        return html_error_page("Source not found", token)
    source.roi_enabled = roi_enabled is not None
    source.roi_x = clamp_roi(roi_x)
    source.roi_y = clamp_roi(roi_y)
    source.roi_w = max(0.01, min(1.0, roi_w))
    source.roi_h = max(0.01, min(1.0, roi_h))
    if source.roi_x + source.roi_w > 1.0:
        source.roi_w = 1.0 - source.roi_x
    if source.roi_y + source.roi_h > 1.0:
        source.roi_h = 1.0 - source.roi_y
    storage.update_source(source)
    runtime.stop_source(source_id)
    if source.enabled:
        runtime.start_source(source_id)
    return redirect_with_token(token)


@app.post("/ui/settings")
def update_settings_ui(
    token: str | None = None,
    cosine_threshold: float = Form(...),
    detection_score_threshold: float = Form(...),
    min_face_width: int = Form(...),
    min_face_height: int = Form(...),
    min_face_area: int = Form(...),
    confirmation_frames: int = Form(...),
):
    ensure_ui_token(token)
    settings = RecognitionSettings(
        cosine_threshold=max(0.0, min(1.0, cosine_threshold)),
        detection_score_threshold=max(0.0, min(1.0, detection_score_threshold)),
        min_face_width=max(1, min_face_width),
        min_face_height=max(1, min_face_height),
        min_face_area=max(1, min_face_area),
        confirmation_frames=max(1, confirmation_frames),
    )
    storage.save_settings(settings)
    return redirect_with_token(token)


@app.post("/ui/sources/{source_id}/start")
def start_source_ui(source_id: str, token: str | None = None):
    ensure_ui_token(token)
    start_source(source_id)
    return redirect_with_token(token)


@app.post("/ui/sources/{source_id}/stop")
def stop_source_ui(source_id: str, token: str | None = None):
    ensure_ui_token(token)
    stop_source(source_id)
    return redirect_with_token(token)


@app.post("/ui/sources/{source_id}/delete")
def delete_source_ui(source_id: str, token: str | None = None):
    ensure_ui_token(token)
    delete_source(source_id)
    return redirect_with_token(token)


@app.get("/health", dependencies=[Depends(require_auth)])
def health() -> dict:
    return {
        "ok": True,
        "people": len(storage.list_people()),
        "sources": len(storage.list_sources()),
        "token_enabled": bool(API_TOKEN),
    }


@app.get("/people", dependencies=[Depends(require_auth)])
def list_people() -> list[dict]:
    return [item.__dict__ for item in storage.list_people()]


@app.post("/people", dependencies=[Depends(require_auth)])
def create_person(name: str = Form(...), info: str = Form(""), photos: list[UploadFile] = File(...)) -> dict:
    if not photos:
        raise HTTPException(status_code=400, detail="At least one photo is required")
    person_id = uuid4().hex
    image_paths = save_person_images(person_id, photos)
    engine = FaceEngine(storage.get_settings())
    embeddings = []
    for path in image_paths:
        embeddings.extend(engine.extract_embeddings_from_bytes((BASE_DIR / path).read_bytes()))
    person = PersonRecord(person_id=person_id, name=name.strip(), info=info.strip(), image_paths=image_paths, embeddings=embeddings)
    people = storage.list_people()
    people.append(person)
    storage.save_people(people)
    return person.__dict__


@app.post("/people/{person_id}/photos", dependencies=[Depends(require_auth)])
def append_person_photos(person_id: str, photos: list[UploadFile] = File(...)) -> dict:
    person = next((item for item in storage.list_people() if item.person_id == person_id), None)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    start_index = len(person.image_paths) + 1
    image_paths = save_person_images(person_id, photos, start_index=start_index)
    engine = FaceEngine(storage.get_settings())
    for path in image_paths:
        person.embeddings.extend(engine.extract_embeddings_from_bytes((BASE_DIR / path).read_bytes()))
    person.image_paths.extend(image_paths)
    storage.update_person(person)
    return person.__dict__


@app.delete("/people/{person_id}", dependencies=[Depends(require_auth)])
def delete_person(person_id: str) -> dict:
    removed = storage.delete_person(person_id)
    if removed is None:
        raise HTTPException(status_code=404, detail="Person not found")
    for relative_path in removed.image_paths:
        image_path = BASE_DIR / relative_path
        if image_path.exists():
            image_path.unlink()
    return {"ok": True}


@app.get("/sources", dependencies=[Depends(require_auth)])
def list_sources() -> list[dict]:
    return [item.__dict__ for item in storage.list_sources()]


@app.get("/sources/status", dependencies=[Depends(require_auth)])
def source_statuses() -> list[dict]:
    return runtime.get_source_statuses()


@app.get("/sources/{source_id}/status", dependencies=[Depends(require_auth)])
def source_status(source_id: str) -> dict:
    statuses = {item["source_id"]: item for item in runtime.get_source_statuses()}
    status = statuses.get(source_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return status


@app.post("/sources", dependencies=[Depends(require_auth)])
def create_source(payload: SourceCreate) -> dict:
    source = storage.add_source(payload.name, payload.url, payload.enabled)
    source.roi_enabled = payload.roi_enabled
    source.roi_x = clamp_roi(payload.roi_x)
    source.roi_y = clamp_roi(payload.roi_y)
    source.roi_w = max(0.01, min(1.0, payload.roi_w))
    source.roi_h = max(0.01, min(1.0, payload.roi_h))
    storage.update_source(source)
    if source.enabled:
        runtime.start_source(source.source_id)
    return source.__dict__


@app.patch("/sources/{source_id}", dependencies=[Depends(require_auth)])
def update_source(source_id: str, payload: SourceUpdate) -> dict:
    source = next((item for item in storage.list_sources() if item.source_id == source_id), None)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    if payload.name is not None:
        source.name = payload.name
    if payload.url is not None:
        source.url = payload.url
    if payload.enabled is not None:
        source.enabled = payload.enabled
    if payload.roi_enabled is not None:
        source.roi_enabled = payload.roi_enabled
    if payload.roi_x is not None:
        source.roi_x = clamp_roi(payload.roi_x)
    if payload.roi_y is not None:
        source.roi_y = clamp_roi(payload.roi_y)
    if payload.roi_w is not None:
        source.roi_w = max(0.01, min(1.0, payload.roi_w))
    if payload.roi_h is not None:
        source.roi_h = max(0.01, min(1.0, payload.roi_h))
    storage.update_source(source)
    runtime.stop_source(source_id)
    if source.enabled:
        runtime.start_source(source_id)
    return source.__dict__


@app.post("/sources/{source_id}/start", dependencies=[Depends(require_auth)])
def start_source(source_id: str) -> dict:
    runtime.start_source(source_id)
    return {"ok": True}


@app.post("/sources/{source_id}/stop", dependencies=[Depends(require_auth)])
def stop_source(source_id: str) -> dict:
    runtime.stop_source(source_id)
    return {"ok": True}


@app.delete("/sources/{source_id}", dependencies=[Depends(require_auth)])
def delete_source(source_id: str) -> dict:
    runtime.stop_source(source_id)
    removed = storage.delete_source(source_id)
    if removed is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return {"ok": True}


@app.get("/events", dependencies=[Depends(require_auth)])
def list_events() -> list[dict]:
    return [item.__dict__ for item in storage.list_events()]


@app.get("/events/latest", dependencies=[Depends(require_auth)])
def latest_event() -> dict:
    events = storage.list_events()
    return events[-1].__dict__ if events else {"event_id": None}
