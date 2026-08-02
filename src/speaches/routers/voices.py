import logging
import re
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


class CreateVoiceResponse(BaseModel):
    id: str
    name: str
    path: str


def _is_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == _WAV_RIFF_MAGIC and data[8:12] == _WAV_WAVE_MAGIC


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

    if file.content_type != "audio/wav" and not _is_wav(raw_bytes):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="only WAV samples are supported",
        )

    CLONE_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    voice_path = CLONE_VOICES_DIR / f"{voice_id}.wav"
    if voice_path.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"voice '{voice_id}' already exists",
        )

    voice_path.write_bytes(raw_bytes)
    logger.info(f"Saved voice '{voice_id}' to {voice_path}")

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "id": voice_id,
            "name": voice_id,
            "path": str(voice_path),
        },
    )
