from collections.abc import Generator
import logging
import pathlib
import threading
import time
from typing import TYPE_CHECKING, Any, cast

import huggingface_hub
import numpy as np
from pydantic import BaseModel, computed_field

from speaches.api_types import Model
from speaches.audio import Audio
from speaches.executors.chatterbox import CLONE_VOICES_DIR
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
    # Lazy import: probe for f5_tts via metadata only (no module execution) so
    # the heavy native deps (torch + the F5TTS pipeline) are not pulled into
    # RSS at process startup. The heavy `from f5_tts.api import F5TTS` is
    # deferred to `_load_fn`, which runs only when an F5 model actually loads.
    import importlib.util

    if importlib.util.find_spec("f5_tts") is None:
        raise ImportError("f5_tts not installed")

    import os

    # Imported before the F5TTS pipeline so the OpenMP-backed torch ops it
    # brings in pick up the pinned thread count from the outset, matching the
    # rationale in the chatterbox executor.
    import torch

    torch.set_num_threads(os.cpu_count() or 1)

    F5_TTS_AVAILABLE = True
except ImportError:
    F5_TTS_AVAILABLE = False

if TYPE_CHECKING:
    # Type-only import so the deferred `"F5TTS"` annotations below resolve for
    # static analysis. Never imported at runtime (lazy: see `_load_fn`).
    from f5_tts.api import F5TTS

SAMPLE_RATE = 24000
LIBRARY_NAME = "f5-tts"
TASK_NAME_TAG = "text-to-speech"

# Cloned voice samples are global across all cloning executors. Imported from
# chatterbox (the canonical source) to avoid drift between the two modules.
# (Re-defined here only as a re-export for readability; the value lives in
# chatterbox.py.)


logger = logging.getLogger(__name__)


class F5ModelVoice(BaseModel):
    name: str

    @computed_field
    @property
    def id(self) -> str:
        return self.name


class F5Model(Model):
    sample_rate: int
    voices: list[F5ModelVoice]


# F5-TTS is clone-only: there is no preset/built-in voice, so unlike chatterbox
# there is no "default" entry. The voices list for a model is populated entirely
# from clone .wav files in CLONE_VOICES_DIR; a user without an uploaded clone
# gets a clear ValueError at generate time (F5 cannot synthesize without a
# reference clip).
VOICES: list[F5ModelVoice] = []

KNOWN_MODELS: dict[str, list[str]] = {
    # Card metadata: license cc-by-nc-4.0, pipeline_tag text-to-speech,
    # library_name f5-tts, trained on Emilia-Dataset (multilingual, en + zh).
    "SWivid/F5-TTS": ["en", "zh"],
}

hf_model_filter = HfModelFilter(library_name=LIBRARY_NAME)


class F5ModelRegistry(ModelRegistry):
    def list_remote_models(self) -> Generator[F5Model]:
        models = huggingface_hub.list_models(**self.hf_model_filter.list_model_kwargs(), cardData=True)
        for model in models:
            if model.created_at is None or model.card_data is None:
                continue
            yield F5Model(
                id=model.id,
                created=int(model.created_at.timestamp()),
                owned_by=model.id.split("/")[0],
                language=extract_language_list(model.card_data),
                task=TASK_NAME_TAG,
                sample_rate=SAMPLE_RATE,
                voices=VOICES,
            )

    def list_local_models(self) -> Generator[F5Model]:
        # Clone voices (one .wav per file in CLONE_VOICES_DIR) are attached to
        # every F5 model, mirroring how the chatterbox executor surfaces clones.
        # F5 is clone-only, so this is the only source of voices for the model.
        clone_voices = (
            [F5ModelVoice(name=f.stem) for f in sorted(CLONE_VOICES_DIR.glob("*.wav"))]
            if CLONE_VOICES_DIR.exists()
            else []
        )
        all_voices = [*VOICES, *clone_voices]

        cached_model_repos_info = get_cached_model_repos_info()
        seen_ids: set[str] = set()
        for cached_repo_info in cached_model_repos_info:
            model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
            if model_card_data is not None and self.hf_model_filter.passes_filter(
                cached_repo_info.repo_id, model_card_data
            ):
                seen_ids.add(cached_repo_info.repo_id)
                yield F5Model(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    language=extract_language_list(model_card_data),
                    task=TASK_NAME_TAG,
                    sample_rate=SAMPLE_RATE,
                    voices=all_voices,
                )
        for cached_repo_info in cached_model_repos_info:
            if cached_repo_info.repo_id in seen_ids:
                continue
            if cached_repo_info.repo_id in KNOWN_MODELS:
                seen_ids.add(cached_repo_info.repo_id)
                yield F5Model(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    language=KNOWN_MODELS[cached_repo_info.repo_id],
                    task=TASK_NAME_TAG,
                    sample_rate=SAMPLE_RATE,
                    voices=all_voices,
                )

    def get_model_files(self, model_id: str) -> None:
        # The base checkpoint (F5TTS_v1_Base/model_1250000.safetensors) is the
        # canonical F5-TTS weights file; its presence confirms the repo is cached.
        huggingface_hub.hf_hub_download(
            repo_id=model_id,
            filename="F5TTS_v1_Base/model_1250000.safetensors",
            local_files_only=True,
        )

    def download_model_files(self, model_id: str) -> None:
        huggingface_hub.snapshot_download(repo_id=model_id, repo_type="model")


f5_model_registry = F5ModelRegistry(hf_model_filter=hf_model_filter)


if F5_TTS_AVAILABLE:

    class F5ModelManager(BaseModelManager["F5TTS"]):
        def __init__(self, ttl: int) -> None:
            super().__init__(ttl)
            self._inference_lock = threading.Lock()

        def _load_fn(self, model_id: str) -> "F5TTS":  # noqa: ARG002
            # model_id is unused: F5TTS downloads its own weights (F5TTS_v1_Base
            # from SWivid/F5-TTS, plus the charactr/vocos-mel-24khz vocoder) on
            # construction, so there is no per-repo from_pretrained to dispatch.
            from f5_tts.api import F5TTS  # lazy: only imported when a model loads

            return F5TTS(model="F5TTS_v1_Base", device="cpu")

        def _clone_path_for_voice(self, voice: str) -> pathlib.Path | None:
            clone_path = CLONE_VOICES_DIR / f"{voice}.wav"
            return clone_path if clone_path.exists() else None

        @traced_generator()
        def handle_speech_request(
            self,
            request: SpeechRequest,
            **_kwargs,
        ) -> SpeechResponse:
            clone_path = self._clone_path_for_voice(request.voice)
            if clone_path is None:
                # F5-TTS is clone-only: it cannot synthesize without a reference
                # clip. Fail clearly rather than crash inside the library.
                msg = (
                    f"Voice '{request.voice}' is not supported. F5-TTS is clone-only: "
                    f"upload a voice sample first (no clone file at {clone_path})."
                )
                raise ValueError(msg)

            text = request.text.strip()
            if not text:
                return

            with self._inference_lock, self.load_model(request.model) as tts:
                tts_any = cast("Any", tts)
                start = time.perf_counter()
                # ref_text="" lets F5 ASR-transcribe the reference clip itself
                # (via its internal Whisper pipeline) — we don't carry clone
                # transcripts, so rely on F5's transcription. This is slower and
                # nondeterministic across runs, but matches the executor contract.
                wav, sr, _spec = tts_any.infer(
                    ref_file=str(clone_path),
                    ref_text="",
                    gen_text=request.text,
                )
                # F5 returns a flat 1-D float32 numpy array at sr (24000). Wrap
                # defensively with np.asarray + int(sr) for type safety.
                yield Audio(np.asarray(wav, dtype=np.float32), sample_rate=int(sr))

            logger.info(f"Generated audio for {len(request.text)} characters in {time.perf_counter() - start}s")
