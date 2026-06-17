import os
import re
import uuid
import shutil
import logging
from flask import Flask, request, jsonify, send_from_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")

TEMP_DIR = "/tmp/transcriber"
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")

# Lazy-load the whisper model
_model = None

def get_model():
    global _model
    if _model is None:
        logger.info(f"Loading Whisper model: {MODEL_SIZE}")
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        logger.info("Model loaded successfully")
    return _model


def is_valid_youtube_url(url: str) -> bool:
    pattern = r"^(https?://)?(www\.)?(youtube\.com/(watch\?v=|shorts/)|youtu\.be/)[a-zA-Z0-9_-]+"
    return bool(re.match(pattern, url))


def download_audio(url: str, output_dir: str) -> str:
    """Download audio from YouTube video using yt-dlp."""
    import yt_dlp

    output_path = os.path.join(output_dir, "audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "Unknown")

    audio_file = os.path.join(output_dir, "audio.mp3")
    if not os.path.exists(audio_file):
        raise FileNotFoundError("Failed to download audio")

    return audio_file, title


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using faster-whisper."""
    model = get_model()
    segments, info = model.transcribe(audio_path, beam_size=5)

    logger.info(f"Detected language: {info.language} (probability {info.language_probability:.2f})")

    full_text = []
    for segment in segments:
        full_text.append(segment.text.strip())

    return " ".join(full_text)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "URL is required"}), 400

    url = data["url"].strip()
    if not is_valid_youtube_url(url):
        return jsonify({"error": "Invalid YouTube URL"}), 400

    job_dir = os.path.join(TEMP_DIR, str(uuid.uuid4()))
    os.makedirs(job_dir, exist_ok=True)

    try:
        logger.info(f"Downloading audio from: {url}")
        audio_path, title = download_audio(url, job_dir)

        logger.info("Starting transcription...")
        text = transcribe_audio(audio_path)

        logger.info("Transcription complete")
        return jsonify({"title": title, "text": text})

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

    finally:
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Pre-load model on startup
    get_model()
    app.run(host="0.0.0.0", port=8000, debug=False)
