from collections.abc import Generator
import logging
import pathlib
import threading
import time
from typing import Any, cast

import huggingface_hub
import numpy as np
from pydantic import BaseModel, computed_field

from speaches.api_types import Model
from speaches.audio import Audio
from speaches.executors.shared.base_model_manager import BaseModelManager
from speaches.executors.shared.handler_protocol import SpeechRequest, SpeechResponse
from speaches.hf_utils import (
    HfModelFilter,
    extract_language_list,
    get_cached_model_repos_info,
    get_model_card_data_from_cached_repo_info,
)
from speaches.model_registry import ModelRegistry
from speaches.tracing import traced_generator

try:
    from chatterbox.tts import ChatterboxTTS
    import torch  # noqa: F401 -- must be imported before ctranslate2 to avoid OpenMP segfault

    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False

SAMPLE_RATE = 24000
LIBRARY_NAME = "chatterbox"
TASK_NAME_TAG = "text-to-speech"

# Chatterbox is cloning-only: there is no fixed preset voice list. Real voices come from
# the clone-sample directory (~/.cache/speaches/voices/*.wav).
PREDEFINED_VOICE_NAMES: list[str] = []

# Directory where user-supplied voice-cloning samples live (one .wav file per voice).
CLONE_VOICES_DIR = pathlib.Path.home() / ".cache" / "speaches" / "voices"

logger = logging.getLogger(__name__)


class ChatterboxModelVoice(BaseModel):
    name: str

    @computed_field
    @property
    def id(self) -> str:
        return self.name


VOICES: list[ChatterboxModelVoice] = []


class ChatterboxModel(Model):
    sample_rate: int
    voices: list[ChatterboxModelVoice]


KNOWN_MODELS: dict[str, list[str]] = {
    "ResembleAI/chatterbox": ["en"],
}

hf_model_filter = HfModelFilter(
    library_name=LIBRARY_NAME,
    # NOTE: ResembleAI/chatterbox has no pipeline_tag set on HuggingFace,
    # so we only filter by library_name to ensure discovery works
)


class ChatterboxModelRegistry(ModelRegistry):
    def list_remote_models(self) -> Generator[ChatterboxModel]:
        models = huggingface_hub.list_models(**self.hf_model_filter.list_model_kwargs(), cardData=True)
        for model in models:
            if model.created_at is None or model.card_data is None:
                continue
            yield ChatterboxModel(
                id=model.id,
                created=int(model.created_at.timestamp()),
                owned_by=model.id.split("/")[0],
                language=extract_language_list(model.card_data),
                task=TASK_NAME_TAG,
                sample_rate=SAMPLE_RATE,
                voices=VOICES,
            )

    def list_local_models(self) -> Generator[ChatterboxModel]:
        cached_model_repos_info = get_cached_model_repos_info()
        seen_ids: set[str] = set()
        for cached_repo_info in cached_model_repos_info:
            model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
            if model_card_data is not None and self.hf_model_filter.passes_filter(
                cached_repo_info.repo_id, model_card_data
            ):
                seen_ids.add(cached_repo_info.repo_id)
                yield ChatterboxModel(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    language=extract_language_list(model_card_data),
                    task=TASK_NAME_TAG,
                    sample_rate=SAMPLE_RATE,
                    voices=VOICES,
                )
        for cached_repo_info in cached_model_repos_info:
            if cached_repo_info.repo_id in seen_ids:
                continue
            if cached_repo_info.repo_id in KNOWN_MODELS:
                seen_ids.add(cached_repo_info.repo_id)
                yield ChatterboxModel(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    language=KNOWN_MODELS[cached_repo_info.repo_id],
                    task=TASK_NAME_TAG,
                    sample_rate=SAMPLE_RATE,
                    voices=VOICES,
                )
        # Expose cloned voices (one .wav file per voice in CLONE_VOICES_DIR) as available models so
        # they appear in /v1/models automatically.
        if CLONE_VOICES_DIR.exists():
            for voice_file in CLONE_VOICES_DIR.glob("*.wav"):
                voice_name = voice_file.stem
                yield ChatterboxModel(
                    id=voice_name,
                    created=int(voice_file.stat().st_mtime),
                    owned_by="speaches",
                    language=[],
                    task=TASK_NAME_TAG,
                    sample_rate=SAMPLE_RATE,
                    voices=[ChatterboxModelVoice(name=voice_name)],
                )

    def get_model_files(self, model_id: str) -> None:
        huggingface_hub.hf_hub_download(
            repo_id=model_id,
            filename="tts_b6369a24.safetensors",
            local_files_only=True,
        )

    def download_model_files(self, model_id: str) -> None:
        huggingface_hub.snapshot_download(repo_id=model_id, repo_type="model")


chatterbox_model_registry = ChatterboxModelRegistry(hf_model_filter=hf_model_filter)


if CHATTERBOX_AVAILABLE:

    class ChatterboxModelManager(BaseModelManager["ChatterboxTTS"]):
        def __init__(self, ttl: int) -> None:
            super().__init__(ttl)
            self._inference_lock = threading.Lock()

        def _load_fn(self, model_id: str) -> "ChatterboxTTS":  # noqa: ARG002
            return ChatterboxTTS.from_pretrained(device="cpu")

        def _clone_path_for_voice(self, voice: str) -> pathlib.Path | None:
            clone_path = CLONE_VOICES_DIR / f"{voice}.wav"
            if clone_path.exists():
                return clone_path
            return None

        @traced_generator()
        def handle_speech_request(
            self,
            request: SpeechRequest,
            **_kwargs,
        ) -> SpeechResponse:
            clone_path = self._clone_path_for_voice(request.voice)
            if clone_path is None and request.voice not in PREDEFINED_VOICE_NAMES:
                msg = f"Voice '{request.voice}' is not supported. No preset voices and no matching clone sample in {CLONE_VOICES_DIR}."
                raise ValueError(msg)

            text = request.text.strip()
            if not text:
                return

            with self._inference_lock, self.load_model(request.model) as tts:
                tts_any = cast("Any", tts)
                start = time.perf_counter()
                if clone_path is not None:
                    audio, sr = tts_any.generate(request.text, audio_prompt_path=str(clone_path))
                else:
                    audio, sr = tts_any.generate(request.text)
                yield Audio(audio.cpu().numpy().astype(np.float32), sample_rate=sr)

            logger.info(f"Generated audio for {len(request.text)} characters in {time.perf_counter() - start}s")
