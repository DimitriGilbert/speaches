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
    import os

    # Pin the thread count before the transformers import so torch's CPU-bound
    # CSM autoregressive sampling picks up the full core count from the outset,
    # matching the rationale in the chatterbox executor.
    import torch

    torch.set_num_threads(os.cpu_count() or 1)
    # Also enable inter-op parallelism so transformers' `generate` can run
    # independent graph ops across cores (intra-op above only covers matmul).
    # Can only be called once per process and before any parallel work begins;
    # guard against it having already been initialized elsewhere.
    import contextlib

    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(max((os.cpu_count() or 1), 1))

    from transformers import AutoProcessor, CsmForConditionalGeneration

    CSM_AVAILABLE = True
except ImportError:
    CSM_AVAILABLE = False

SAMPLE_RATE = 24000
# CSM-1b ships inside `transformers` (the HF card's library_name is "transformers").
# That library filter is broad, so we additionally narrow by model_name="csm-1b"
# to ensure this registry only claims sesame/csm-1b out of the many cached
# transformers repos a typical install has.
LIBRARY_NAME = "transformers"
TASK_NAME_TAG = "text-to-speech"
MODEL_ID = "sesame/csm-1b"

# Cloned voice samples are global across all cloning executors. Imported from
# chatterbox (the canonical source) to avoid drift between the modules, mirroring
# how the f5_tts executor does it.

logger = logging.getLogger(__name__)


class CsmModelVoice(BaseModel):
    name: str

    @computed_field
    @property
    def id(self) -> str:
        return self.name


class CsmModel(Model):
    sample_rate: int
    voices: list[CsmModelVoice]


# CSM-1b is clone-only: there is no preset/built-in voice. The voices list for
# the model is populated entirely from clone .wav files in CLONE_VOICES_DIR; a
# user without an uploaded clone gets a clear ValueError at generate time
# (CSM cannot synthesize without a reference clip + transcript).
VOICES: list[CsmModelVoice] = []

KNOWN_MODELS: dict[str, list[str]] = {
    "sesame/csm-1b": ["en"],
}

hf_model_filter = HfModelFilter(library_name=LIBRARY_NAME, model_name="csm-1b")


class CsmModelRegistry(ModelRegistry):
    def list_remote_models(self) -> Generator[CsmModel]:
        models = huggingface_hub.list_models(**self.hf_model_filter.list_model_kwargs(), cardData=True)
        for model in models:
            if model.created_at is None or model.card_data is None:
                continue
            yield CsmModel(
                id=model.id,
                created=int(model.created_at.timestamp()),
                owned_by=model.id.split("/")[0],
                language=extract_language_list(model.card_data),
                task=TASK_NAME_TAG,
                sample_rate=SAMPLE_RATE,
                voices=VOICES,
            )

    def list_local_models(self) -> Generator[CsmModel]:
        # Clone voices (one .wav per file in CLONE_VOICES_DIR) are attached to
        # the csm model, mirroring how the chatterbox/f5 executors surface clones.
        # CSM is clone-only, so this is the only source of voices for the model.
        clone_voices = (
            [CsmModelVoice(name=f.stem) for f in sorted(CLONE_VOICES_DIR.glob("*.wav"))]
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
                yield CsmModel(
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
                yield CsmModel(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    language=KNOWN_MODELS[cached_repo_info.repo_id],
                    task=TASK_NAME_TAG,
                    sample_rate=SAMPLE_RATE,
                    voices=all_voices,
                )

    def get_model_files(self, model_id: str) -> None:
        # The CSM card ships a config.json; its presence confirms the repo is cached.
        huggingface_hub.hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            local_files_only=True,
        )

    def download_model_files(self, model_id: str) -> None:
        huggingface_hub.snapshot_download(repo_id=model_id, repo_type="model")


csm_model_registry = CsmModelRegistry(hf_model_filter=hf_model_filter)


if CSM_AVAILABLE:

    class CsmModelManager(BaseModelManager["CsmForConditionalGeneration"]):
        def __init__(self, ttl: int) -> None:
            super().__init__(ttl)
            self._inference_lock = threading.Lock()
            # Lazily loaded alongside the model in _load_fn; kept on the manager
            # so handle_speech_request can build the conversation chat template.
            self._processor: Any = None
            # ref path -> transcript. Avoids re-transcribing the same clone on
            # every request (the whisper pass is the slow part of cloning here).
            self._transcription_cache: dict[str, str] = {}
            # Lazily loaded faster-whisper model used to transcribe clone refs.
            self._whisper: Any = None

        def _load_fn(self, model_id: str) -> "CsmForConditionalGeneration":  # noqa: ARG002
            # model_id is unused: CSM only has one repo (sesame/csm-1b), so there
            # is no per-repo dispatch. Load the processor and model from the same
            # repo id; stash the processor on the manager for chat-template use.
            self._processor = AutoProcessor.from_pretrained(MODEL_ID)
            # `from_pretrained` with device_map is typed as a union over the
            # tuple-dispatch path; cast to the declared model type to satisfy
            # pyrefly (the actual runtime return is the single CSM model).
            model = cast(
                "CsmForConditionalGeneration",
                CsmForConditionalGeneration.from_pretrained(MODEL_ID, device_map="cpu"),
            )
            model.eval()
            return model

        def _clone_path_for_voice(self, voice: str) -> pathlib.Path | None:
            clone_path = CLONE_VOICES_DIR / f"{voice}.wav"
            return clone_path if clone_path.exists() else None

        def _transcribe_ref(self, clone_path: pathlib.Path) -> str:
            # CSM needs the reference clip's transcript to align the clone. We
            # don't store transcripts, so transcribe at request time with
            # faster-whisper (a core dep). Cache by path so repeated requests
            # don't re-transcribe. The distil small.en model is enough for a
            # short reference transcript and keeps the cost low on CPU.
            cache_key = str(clone_path)
            if cache_key in self._transcription_cache:
                return self._transcription_cache[cache_key]
            if self._whisper is None:
                from faster_whisper import WhisperModel

                self._whisper = WhisperModel(
                    "Systran/faster-distil-whisper-small.en", device="cpu", compute_type="int8"
                )
            segments, _info = self._whisper.transcribe(str(clone_path))
            text = " ".join(s.text for s in segments).strip()
            self._transcription_cache[cache_key] = text
            return text

        def _load_ref_audio_24k(self, clone_path: pathlib.Path) -> np.ndarray:
            # CSM requires a 24kHz mono reference. Clone wavs may be stored at
            # arbitrary rates (the upload endpoint transcodes to 16kHz), so
            # resample to 24kHz and downmix to mono. Returns a numpy float32
            # 1-D array, which transformers' load_audio accepts (it rejects
            # torch tensors, only taking a URL/path string or numpy array).
            import torchaudio

            wav, sr = torchaudio.load(str(clone_path))
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav.numpy().astype(np.float32).reshape(-1)

        @traced_generator()
        def handle_speech_request(
            self,
            request: SpeechRequest,
            **_kwargs,
        ) -> SpeechResponse:
            clone_path = self._clone_path_for_voice(request.voice)
            if clone_path is None:
                # CSM-1b is clone-only: it cannot synthesize without a reference
                # clip. Fail clearly rather than crash inside the library.
                msg = (
                    f"Voice '{request.voice}' is not supported. CSM-1b is clone-only: "
                    f"upload a voice sample first (no clone file at {clone_path})."
                )
                raise ValueError(msg)

            text = request.text.strip()
            if not text:
                return

            with self._inference_lock, self.load_model(request.model) as model:
                if self._processor is None:
                    # Loaded in _load_fn; guard defensively in case the manager
                    # state was reset between loads.
                    msg = "CSM processor not loaded"
                    raise RuntimeError(msg)
                start = time.perf_counter()
                ref_text = self._transcribe_ref(clone_path)
                ref_audio = self._load_ref_audio_24k(clone_path)
                # Cloning is conversation-based via the chat template. The first
                # turn pairs the reference transcript with the reference audio
                # (speaker "0" = the cloned voice); the final text-only turn is
                # what gets synthesized.
                conversation = [
                    {
                        "role": "0",
                        "content": [
                            {"type": "text", "text": ref_text},
                            {"type": "audio", "path": ref_audio},
                        ],
                    },
                    {"role": "0", "content": [{"type": "text", "text": request.text}]},
                ]
                inputs = self._processor.apply_chat_template(conversation, tokenize=True, return_dict=True)
                model_any = cast("Any", model)
                # model.generate returns a list[torch.FloatTensor], one per batch
                # item (batch=1 here). Each item is a flat 1-D float tensor at
                # 24kHz mono. The sample rate is the processor's default and is
                # not a model attribute, so it's hardcoded as SAMPLE_RATE above.
                audio_outputs = model_any.generate(**inputs, output_audio=True)
                audio_tensor = audio_outputs[0]
                yield Audio(audio_tensor.cpu().numpy().astype(np.float32).reshape(-1), sample_rate=SAMPLE_RATE)

            logger.info(f"Generated audio for {len(request.text)} characters in {time.perf_counter() - start}s")
