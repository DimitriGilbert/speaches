import logging
import re
import subprocess
from typing import Annotated

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from speaches.executors.chatterbox import CLONE_VOICES_DIR

logger = logging.getLogger(__name__)

router = APIRouter(tags=["text-to-speech"])

# Voice ids are restricted to lowercase letters, digits, hyphen, and underscore.
# This rejects empty names, "/", "..", spaces, and any unicode — making path
# traversal impossible (the id is used directly as a filename stem).
VOICE_NAME_PATTERN = re.compile(r"[a-z0-9_-]+")

MAX_SAMPLE_SIZE = 10 * 1024 * 1024

# WAV files start with the ASCII bytes "RIFF" at offset 0 and "WAVE" at offset 8.
_WAV_RIFF_MAGIC = b"RIFF"
_WAV_WAVE_MAGIC = b"WAVE"

# Clone samples are stored as 16 kHz mono wav — a clean reference for the
# zero-shot cloning model regardless of the uploaded format (mic recordings
# arrive as webm/opus; uploads may be mp3, m4a, etc.).
_CLONE_SAMPLE_RATE = 16000


class CreateVoiceResponse(BaseModel):
    id: str
    name: str
    path: str


def _is_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == _WAV_RIFF_MAGIC and data[8:12] == _WAV_WAVE_MAGIC


def _transcode_to_wav(src: bytes, dst_path_str: str) -> None:
    """Transcode arbitrary audio bytes to 16 kHz mono wav via ffmpeg.

    Reads from stdin and writes to dst_path_str so no temp file is needed.
    Raises HTTPException(422) if ffmpeg is missing or cannot decode the input
    (e.g. a non-audio upload or an unsupported codec).
    """
    try:
        subprocess.run(
            [  # noqa: S607 -- ffmpeg is on PATH in the base image (apt-installed)
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-ar",
                str(_CLONE_SAMPLE_RATE),
                "-ac",
                "1",
                "-y",
                dst_path_str,
            ],
            input=src,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ffmpeg is not available to transcode the sample",
        ) from err
    except subprocess.CalledProcessError as err:
        stderr = err.stderr.decode(errors="replace").strip() if err.stderr else ""
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"could not decode the audio sample{f': {stderr[:200]}' if stderr else ''}",
        ) from err


@router.post(
    "/v1/audio/voices",
    response_model=CreateVoiceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_voice(
    name: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
) -> JSONResponse:
    voice_id = name.lower()
    if not VOICE_NAME_PATTERN.fullmatch(voice_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid voice name; use lowercase letters, digits, hyphen, underscore",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="sample is empty")

    if len(raw_bytes) > MAX_SAMPLE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="sample too large (max 10 MB)",
        )

    CLONE_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    voice_path = CLONE_VOICES_DIR / f"{voice_id}.wav"
    if voice_path.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"voice '{voice_id}' already exists",
        )

    # Mic recordings arrive as webm/opus and uploads may be mp3/m4a/etc. Accept
    # any format ffmpeg can decode; wav is saved as-is, everything else is
    # transcoded to 16 kHz mono wav (the clone-reference format).
    if _is_wav(raw_bytes):
        voice_path.write_bytes(raw_bytes)
    else:
        _transcode_to_wav(raw_bytes, str(voice_path))
    logger.info(f"Saved voice '{voice_id}' to {voice_path}")

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "id": voice_id,
            "name": voice_id,
            "path": str(voice_path),
        },
    )
