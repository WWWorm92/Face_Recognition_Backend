# Face Recognition Backend

Backend для распознавания лиц и интеграции с Home Assistant.

## Что внутри

- `service/` - FastAPI API, web UI и фоновые воркеры распознавания;
- `deploy/` - Docker Compose, nginx и systemd шаблоны.

## Возможности

- несколько фото на одного человека;
- события распознавания по потокам;
- сохранение snapshot на каждое событие;
- настройки точности через web UI;
- web UI для управления людьми и источниками.

## Запуск локально

```bash
pip install -r requirements.txt
uvicorn service.app:app --host 0.0.0.0 --port 8787
```

Web UI:

- `http://localhost:8787/`
- `http://localhost:8787/docs`

Snapshots событий сохраняются в `service_data/snapshots/` и доступны через backend.

## Запуск на сервере

Отредактируй `deploy/remote-server-docker-compose.yml`, задай `FACE_API_TOKEN`, затем:

```bash
docker-compose -f deploy/remote-server-docker-compose.yml up -d --build
```
