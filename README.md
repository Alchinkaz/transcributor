# Transcributor

Минималистичный веб-сервис для транскрибации YouTube видео с использованием OpenAI Whisper.

## Как это работает

1. Вставьте ссылку на YouTube видео
2. Нажмите «Расшифровать»
3. Получите полный текст транскрибации

## Технологии

- **Backend**: Python / Flask
- **Транскрибация**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (быстрая реализация OpenAI Whisper)
- **Скачивание видео**: [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- **Аудио обработка**: FFmpeg
- **Деплой**: Docker Compose

## Запуск

```bash
docker compose up --build -d
```

Сервис будет доступен на `http://localhost:8000`

## Конфигурация

Через переменные окружения в `docker-compose.yaml`:

| Переменная | По умолчанию | Описание |
|---|---|---|
| `WHISPER_MODEL` | `base` | Размер модели: `tiny`, `base`, `small`, `medium`, `large-v3` |

Чем больше модель — тем точнее транскрибация, но больше потребление RAM и время обработки.

## Структура проекта

```
├── app/
│   ├── main.py              # Flask backend
│   ├── requirements.txt     # Python dependencies
│   └── static/
│       └── index.html        # Frontend
├── Dockerfile
├── docker-compose.yaml
└── README.md
```
