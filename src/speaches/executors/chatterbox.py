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
    import os

    from chatterbox import ChatterboxMultilingualTTS
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    # Imported before ctranslate2 to avoid an OpenMP segfault; also used below
    # to pin the thread count for the CPU-bound T3/VE autoregressive sampling.
    import torch

    # Torch defaults to half the cores in some builds, which roughly halves TTS
    # throughput — use all available cores so generation isn't needlessly slow.
    torch.set_num_threads(os.cpu_count() or 1)

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
    # Lower-latency 350M variant of the base model — same clone API, faster CPU
    # inference. Falls back to a plain ChatterboxTTS load if the Turbo class is
    # unavailable (older chatterbox-tts).
    "ResembleAI/chatterbox-turbo": ["en"],
    # 23+ language multilingual variant. Requires a language_id at generate()
    # time; the executor passes the model's first listed language as default.
    "ResembleAI/chatterbox-multilingual": [
        "en",
        "zh",
        "ja",
        "ko",
        "fr",
        "de",
        "es",
        "it",
        "pt",
        "ru",
        "ar",
        "hi",
    ],
}

# Chatterbox model ids that load the Turbo variant (smaller, faster on CPU).
# Matched by substring so user-pushed fine-tunes (e.g. ".../chatterbox-turbo-v2")
# resolve correctly without editing this table.
TURBO_MODEL_IDS = {"ResembleAI/chatterbox-turbo"}
MULTILINGUAL_MODEL_IDS = {"ResembleAI/chatterbox-multilingual"}

# Language ids the multilingual model accepts. The executor maps an ISO code
# (or full language name) to the integer the model expects. Source: the model
# card's t3_mtl23ls tokenizer ordering.
MULTILINGUAL_LANGUAGES: dict[str, int] = {
    "en": 0,
    "zh": 1,
    "ja": 2,
    "ko": 3,
    "fr": 4,
    "de": 5,
    "es": 6,
    "it": 7,
    "pt": 8,
    "ru": 9,
    "ar": 10,
    "hi": 11,
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
        # Cloned voices (one .wav per file in CLONE_VOICES_DIR) are attached to
        # every chatterbox model as additional voices — NOT exposed as separate
        # models. The UI derives the voice dropdown from the selected model's
        # voices[], so clones must live under the model the user picks.
        clone_voices = (
            [ChatterboxModelVoice(name=f.stem) for f in sorted(CLONE_VOICES_DIR.glob("*.wav"))]
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
                yield ChatterboxModel(
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
                yield ChatterboxModel(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    language=KNOWN_MODELS[cached_repo_info.repo_id],
                    task=TASK_NAME_TAG,
                    sample_rate=SAMPLE_RATE,
                    voices=all_voices,
                )

    def get_model_files(self, model_id: str) -> None:
        # The T3 (text-to-speech token) weights are present in every Chatterbox
        # variant; checking them is enough to confirm the repo is cached.
        huggingface_hub.hf_hub_download(
            repo_id=model_id,
            filename="t3_cfg.safetensors",
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

        def _load_fn(self, model_id: str) -> "ChatterboxTTS":
            # Dispatch to the right Chatterbox variant by model id. All three
            # share from_pretrained(device=...) and a generate(text, ...,
            # audio_prompt_path=...) clone API, so one manager covers them.
            if model_id in MULTILINGUAL_MODEL_IDS:
                return ChatterboxMultilingualTTS.from_pretrained(device="cpu")
            if model_id in TURBO_MODEL_IDS:
                return ChatterboxTurboTTS.from_pretrained(device="cpu")
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
                # The multilingual variant requires a language_id. The voice id
                # may carry a language hint as "<voice>__<lang>"; otherwise fall
                # back to the model's first listed language.
                if request.model in MULTILINGUAL_MODEL_IDS:
                    lang_id = self._resolve_language(request.voice, request.model)
                    if clone_path is not None:
                        audio = tts_any.generate(text, language_id=lang_id, audio_prompt_path=str(clone_path))
                    else:
                        audio = tts_any.generate(text, language_id=lang_id)
                elif clone_path is not None:
                    audio = tts_any.generate(request.text, audio_prompt_path=str(clone_path))
                else:
                    audio = tts_any.generate(request.text)
                # Chatterbox.generate() returns a single torch.Tensor of shape
                # (1, N) — mono as a 2D array. The sample rate lives on the
                # model as `sr`. Flatten to 1-D float32: speaches' Audio (and
                # the as_bytes/extend paths) expect a 1-D buffer.
                yield Audio(audio.cpu().numpy().astype(np.float32).reshape(-1), sample_rate=tts_any.sr)

            logger.info(f"Generated audio for {len(request.text)} characters in {time.perf_counter() - start}s")

        @staticmethod
        def _resolve_language(voice: str, model_id: str) -> int:
            # Voice ids may embed a language hint as "<voice>__<iso>", e.g.
            # "narrator__de". If present, use it; otherwise default to the
            # model's first listed language (the KNOWN_MODELS entry).
            default_lang = KNOWN_MODELS.get(model_id, ["en"])[0]
            if "__" in voice:
                hint = voice.rsplit("__", 1)[1].lower()
                if hint in MULTILINGUAL_LANGUAGES:
                    return MULTILINGUAL_LANGUAGES[hint]
            return MULTILINGUAL_LANGUAGES.get(default_lang, 0)
