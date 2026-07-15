import os
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from service.engine import FaceEngine
from service.runtime import RuntimeManager
from service.schemas import SourceCreate, SourceUpdate
from service.storage import BASE_DIR, PEOPLE_DIR, PersonRecord, Storage


app = FastAPI(title="Home Assistant Face Recognition", version="0.1.0")
storage = Storage()
runtime = RuntimeManager(storage)
API_TOKEN = os.getenv("FACE_API_TOKEN", "").strip()


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
            body {{ font-family: Arial, sans-serif; background:#111827; color:#f9fafb; margin:0; padding:24px; }}
            h1,h2,h3 {{ margin:0 0 12px; }}
            .grid {{ display:grid; grid-template-columns:1.1fr 1fr; gap:20px; }}
            .row {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
            .card {{ background:#1f2937; border-radius:14px; padding:18px; margin-bottom:18px; }}
            .item {{ border-top:1px solid #374151; padding-top:14px; margin-top:14px; }}
            .muted {{ color:#9ca3af; font-size:14px; }}
            .metric {{ font-size:28px; font-weight:700; }}
            .mono {{ font-family:Consolas, monospace; word-break:break-all; }}
            input, textarea {{ width:100%; padding:10px; border-radius:10px; border:1px solid #374151; background:#111827; color:#f9fafb; box-sizing:border-box; }}
            textarea {{ min-height:88px; resize:vertical; }}
            button {{ padding:10px 14px; border:none; border-radius:10px; background:#2563eb; color:white; cursor:pointer; }}
            button.secondary {{ background:#374151; }}
            button.danger {{ background:#b91c1c; }}
            a {{ color:#93c5fd; }}
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
    statuses = {item["source_id"]: item for item in runtime.get_source_statuses()}
    events = storage.list_events()[-20:][::-1]

    people_html = "".join(
        f"""
        <div class='item'>
          <h3>{person.name}</h3>
          <div class='muted'>{person.info or 'Без дополнительных данных'}</div>
          <div class='row' style='margin-top:10px;'>
            <span>Фото: <b>{len(person.image_paths)}</b></span>
            <span>Эмбеддинги: <b>{len(person.embeddings)}</b></span>
          </div>
          <div class='row' style='margin-top:12px;'>
            <form method='post' action='/ui/people/{person.person_id}/delete?token={token or ""}'>
              <button class='danger' type='submit'>Удалить</button>
            </form>
          </div>
        </div>
        """
        for person in people
    ) or "<div class='muted'>Людей пока нет.</div>"

    sources_html = "".join(
        f"""
        <div class='item'>
          <h3>{source.name}</h3>
          <div class='mono muted'>{source.url}</div>
          <div class='row' style='margin-top:10px;'>
            <span>Статус: <b>{statuses.get(source.source_id, {}).get('status', 'unknown')}</b></span>
            <span>Последний человек: <b>{statuses.get(source.source_id, {}).get('last_person_name', '') or '-'}</b></span>
            <span>Score: <b>{statuses.get(source.source_id, {}).get('last_score', 0.0)}</b></span>
          </div>
          <div class='muted'>Resolved URL: {statuses.get(source.source_id, {}).get('resolved_url', '') or '-'}</div>
          <div class='muted'>Ошибка: {statuses.get(source.source_id, {}).get('last_error', '') or '-'}</div>
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
        f"<div class='item'><b>{event.source_name}</b> | {event.person_name} | score {event.score} | {event.timestamp}</div>"
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
    """
    return html_page("Face Recognition Admin", body)


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
    engine = FaceEngine()
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
    engine = FaceEngine()
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
