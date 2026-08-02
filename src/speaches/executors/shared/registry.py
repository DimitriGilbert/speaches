from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from typing import TYPE_CHECKING, Any

from speaches.config import OrtOptions
from speaches.executors.kokoro import KokoroModelRegistry
from speaches.executors.parakeet import NemoConformerTdtModelRegistry
from speaches.executors.piper import PiperModelRegistry
from speaches.executors.pocket_tts import POCKET_TTS_AVAILABLE, PocketTTSModelRegistry
from speaches.executors.pyannote_diarization import (
    PyannoteDiarizationModelManager,
    PyannoteDiarizationModelRegistry,
    pyannote_diarization_model_registry,
)
from speaches.executors.silero_vad_v5 import SileroVADModelRegistry
from speaches.executors.wespeaker_speaker_embedding import WespeakerSpeakerEmbeddingModelRegistry
from speaches.executors.whisper import WhisperModelRegistry

if TYPE_CHECKING:
    from speaches.config import Config
    from speaches.executors.shared.handler_protocol import SpeechHandler, TranscriptionHandler

from speaches.executors.kokoro import KokoroModelManager, kokoro_model_registry
from speaches.executors.parakeet import ParakeetModelManager, parakeet_model_registry
from speaches.executors.piper import PIPER_AVAILABLE
from speaches.executors.shared.executor import Executor

if PIPER_AVAILABLE:
    from speaches.executors.piper import PiperModelManager, piper_model_registry

if POCKET_TTS_AVAILABLE:
    from speaches.executors.pocket_tts import PocketTTSModelManager, pocket_tts_model_registry

KOKORO_PYTORCH_AVAILABLE = False
try:
    from speaches.executors.kokoro_pytorch import (
        KokoroPytorchModelManager,
        KokoroPytorchModelRegistry,
        kokoro_pytorch_model_registry,
    )

    KOKORO_PYTORCH_AVAILABLE = True
except ImportError:
    pass

QWEN3_TTS_AVAILABLE = False
try:
    from speaches.executors.qwen3_tts import (
        Qwen3TTSModelManager,
        Qwen3TTSModelRegistry,
        qwen3_tts_model_registry,
    )

    QWEN3_TTS_AVAILABLE = True
except ImportError:
    pass

CHATTERBOX_AVAILABLE = False
try:
    from speaches.executors.chatterbox import (
        ChatterboxModelManager,
        ChatterboxModelRegistry,
        chatterbox_model_registry,
    )

    CHATTERBOX_AVAILABLE = True
except ImportError:
    pass

from speaches.executors.silero_vad_v5 import SileroVADModelManager  # noqa: E402
from speaches.executors.wespeaker_speaker_embedding import (  # noqa: E402
    WespeakerSpeakerEmbeddingModelManager,
    wespeaker_speaker_embedding_model_registry,
)
from speaches.executors.whisper import WhisperModelManager, whisper_model_registry  # noqa: E402
from speaches.hf_utils import HfModelFilter  # noqa: E402

logger = logging.getLogger(__name__)


def _cpu_only_ort_opts(base: OrtOptions) -> OrtOptions:
    exclude = set(base.exclude_providers)
    exclude.add("CUDAExecutionProvider")
    return OrtOptions(
        exclude_providers=list(exclude),
        provider_priority={},
        provider_opts={},
        gpu_mem_limit=None,
        enable_cpu_mem_arena=base.enable_cpu_mem_arena,
        enable_mem_pattern=base.enable_mem_pattern,
    )


class ExecutorRegistry:
    def __init__(self, config: Config) -> None:
        self._exit_stack = contextlib.ExitStack()
        self._pinned_models: list[Any] = []
        self._pinned_model_ids: set[str] = set()
        self._tts_cache: dict[str, SpeechHandler] = {}
        self._stt_cache: dict[str, TranscriptionHandler] = {}
        self._tts_resolve_lock: threading.Lock = threading.Lock()
        self._stt_resolve_lock: threading.Lock = threading.Lock()
        gpu_ort_opts = config.unstable_ort_opts.model_copy(update={"gpu_mem_limit": config.gpu_mem_limit})
        cpu_ort_opts = _cpu_only_ort_opts(config.unstable_ort_opts)
        self._whisper_executor = Executor[WhisperModelManager, WhisperModelRegistry](
            name="whisper",
            model_manager=WhisperModelManager(config.stt_model_ttl, config.whisper),
            model_registry=whisper_model_registry,
            task="automatic-speech-recognition",
        )
        self._parakeet_executor = Executor[ParakeetModelManager, NemoConformerTdtModelRegistry](
            name="parakeet",
            model_manager=ParakeetModelManager(config.stt_model_ttl, gpu_ort_opts),
            model_registry=parakeet_model_registry,
            task="automatic-speech-recognition",
        )
        self._piper_executor: Executor | None = None
        if PIPER_AVAILABLE:
            self._piper_executor = Executor[PiperModelManager, PiperModelRegistry](
                name="piper",
                model_manager=PiperModelManager(config.tts_model_ttl, gpu_ort_opts),
                model_registry=piper_model_registry,
                task="text-to-speech",
            )
        self._kokoro_pytorch_executor: Executor | None = None
        if KOKORO_PYTORCH_AVAILABLE:
            self._kokoro_pytorch_executor = Executor[KokoroPytorchModelManager, KokoroPytorchModelRegistry](
                name="kokoro-pytorch",
                model_manager=KokoroPytorchModelManager(config.tts_model_ttl),
                model_registry=kokoro_pytorch_model_registry,
                task="text-to-speech",
            )
        self._pocket_tts_executor: Executor | None = None
        if POCKET_TTS_AVAILABLE:
            self._pocket_tts_executor = Executor[PocketTTSModelManager, PocketTTSModelRegistry](
                name="pocket-tts",
                model_manager=PocketTTSModelManager(config.tts_model_ttl),
                model_registry=pocket_tts_model_registry,
                task="text-to-speech",
            )
        self._qwen3_tts_executor: Executor | None = None
        if QWEN3_TTS_AVAILABLE:
            self._qwen3_tts_executor = Executor[Qwen3TTSModelManager, Qwen3TTSModelRegistry](
                name="qwen3-tts",
                model_manager=Qwen3TTSModelManager(
                    config.tts_model_ttl,
                    load_in_8bit=config.qwen3_tts_load_in_8bit,
                    default_voice=config.qwen3_tts_default_voice,
                    default_design_instruct=config.qwen3_tts_default_design_instruct,
                ),
                model_registry=qwen3_tts_model_registry,
                task="text-to-speech",
            )
        self._chatterbox_executor: Executor | None = None
        if CHATTERBOX_AVAILABLE:
            self._chatterbox_executor = Executor[ChatterboxModelManager, ChatterboxModelRegistry](
                name="chatterbox",
                model_manager=ChatterboxModelManager(config.tts_model_ttl),
                model_registry=chatterbox_model_registry,
                task="text-to-speech",
            )
        self._kokoro_executor = Executor[KokoroModelManager, KokoroModelRegistry](
            name="kokoro",
            model_manager=KokoroModelManager(config.tts_model_ttl, gpu_ort_opts),
            model_registry=kokoro_model_registry,
            task="text-to-speech",
        )
        self._wespeaker_speaker_embedding_executor = Executor[
            WespeakerSpeakerEmbeddingModelManager, WespeakerSpeakerEmbeddingModelRegistry
        ](
            name="wespeaker-speaker-embedding",
            model_manager=WespeakerSpeakerEmbeddingModelManager(0),  # HACK: hardcoded ttl
            model_registry=wespeaker_speaker_embedding_model_registry,
            task="speaker-embedding",
        )
        self._pyannote_diarization_executor = Executor[
            PyannoteDiarizationModelManager, PyannoteDiarizationModelRegistry
        ](
            name="pyannote-diarization",
            model_manager=PyannoteDiarizationModelManager(-1),  # HACK: hardcoded ttl
            model_registry=pyannote_diarization_model_registry,
            task="speaker-diarization",
        )
        self._vad_model_id = config.vad_model
        vad_model_registry = SileroVADModelRegistry(
            hf_model_filter=HfModelFilter(library_name="onnx", task="voice-activity-detection"),
            active_model_id=config.vad_model,
        )
        self._vad_executor = Executor[SileroVADModelManager, SileroVADModelRegistry](
            name="vad",
            model_manager=SileroVADModelManager(config.vad_model_ttl, cpu_ort_opts, vad_model_registry),
            model_registry=vad_model_registry,
            task="voice-activity-detection",
        )

    @property
    def transcription(self) -> tuple[Executor, ...]:
        return (self._whisper_executor, self._parakeet_executor)

    @property
    def translation(self) -> tuple[Executor, ...]:
        return (self._whisper_executor,)

    @property
    def text_to_speech(self) -> tuple[Executor, ...]:
        executors: list[Executor] = []
        if self._piper_executor is not None:
            executors.append(self._piper_executor)
        if self._pocket_tts_executor is not None:
            executors.append(self._pocket_tts_executor)
        if self._kokoro_pytorch_executor is not None:
            executors.append(self._kokoro_pytorch_executor)
        if self._qwen3_tts_executor is not None:
            executors.append(self._qwen3_tts_executor)
        if self._chatterbox_executor is not None:
            executors.append(self._chatterbox_executor)
        executors.append(self._kokoro_executor)
        return tuple(executors)

    @property
    def speaker_embedding(self) -> tuple[Executor, ...]:
        return (self._wespeaker_speaker_embedding_executor,)

    @property
    def diarization(self):  # noqa: ANN201
        return (self._pyannote_diarization_executor,)

    @property
    def vad_model_id(self) -> str:
        return self._vad_model_id

    @property
    def vad(self) -> Executor:
        return self._vad_executor

    def all_executors(self) -> tuple[Executor, ...]:
        executors: list[Executor] = [
            self._whisper_executor,
            self._parakeet_executor,
        ]
        if self._piper_executor is not None:
            executors.append(self._piper_executor)
        if self._pocket_tts_executor is not None:
            executors.append(self._pocket_tts_executor)
        if self._kokoro_pytorch_executor is not None:
            executors.append(self._kokoro_pytorch_executor)
        if self._qwen3_tts_executor is not None:
            executors.append(self._qwen3_tts_executor)
        if self._chatterbox_executor is not None:
            executors.append(self._chatterbox_executor)
        executors.extend(
            [
                self._kokoro_executor,
                self._wespeaker_speaker_embedding_executor,
                self._pyannote_diarization_executor,
                self._vad_executor,
            ]
        )
        return tuple(executors)

    def download_model_by_id(self, model_id: str) -> bool:
        for executor in self.all_executors():
            try:
                local_ids = {m.id for m in executor.model_registry.list_local_models()}
            except (OSError, ValueError):
                local_ids = set()
            if model_id in local_ids:
                return True
            try:
                remote_ids = {m.id for m in executor.model_registry.list_remote_models()}
            except (OSError, ValueError):
                remote_ids = set()
            if model_id in remote_ids:
                return executor.model_registry.download_model_files_if_not_exist(model_id)
        raise ValueError(f"Model '{model_id}' not found")

    def _find_executor_for_model(self, model_id: str) -> Executor | None:
        for executor in self.all_executors():
            try:
                ids = {m.id for m in executor.model_registry.list_local_models()}
                if not ids:
                    ids = {m.id for m in executor.model_registry.list_remote_models()}
            except (OSError, ValueError):
                continue
            if model_id in ids:
                return executor
        return None

    def resolve_tts_model_manager(self, model_id: str) -> SpeechHandler:
        cached = self._tts_cache.get(model_id)
        if cached is not None:
            return cached
        with self._tts_resolve_lock:
            cached = self._tts_cache.get(model_id)
            if cached is not None:
                return cached
            for executor in self.text_to_speech:
                try:
                    model_ids = [m.id for m in executor.model_registry.list_local_models()]
                    if not model_ids:
                        model_ids = [m.id for m in executor.model_registry.list_remote_models()]
                except (OSError, ValueError):
                    logger.debug(f"Failed to list models for executor '{executor.name}', skipping")
                    continue
                if model_id in model_ids:
                    self._tts_cache[model_id] = executor.model_manager
                    return executor.model_manager
            raise ValueError(f"No TTS executor found for model '{model_id}'")

    def resolve_stt_model_manager(self, model_id: str) -> TranscriptionHandler:
        cached = self._stt_cache.get(model_id)
        if cached is not None:
            return cached
        with self._stt_resolve_lock:
            cached = self._stt_cache.get(model_id)
            if cached is not None:
                return cached
            for executor in self.transcription:
                try:
                    model_ids = [m.id for m in executor.model_registry.list_local_models()]
                    if not model_ids:
                        model_ids = [m.id for m in executor.model_registry.list_remote_models()]
                except (OSError, ValueError):
                    logger.debug(f"Failed to list models for executor '{executor.name}', skipping")
                    continue
                if model_id in model_ids:
                    self._stt_cache[model_id] = executor.model_manager
                    return executor.model_manager
            raise ValueError(f"No STT executor found for model '{model_id}'")

    async def warmup_model(self, model_id: str) -> None:
        if model_id in self._pinned_model_ids:
            return
        executor = self._find_executor_for_model(model_id)
        if executor is None:
            raise ValueError(f"Model '{model_id}' not found")
        disposable = await asyncio.to_thread(executor.model_manager.load_model, model_id)
        model = self._exit_stack.enter_context(disposable)
        self._pinned_models.append(model)
        self._pinned_model_ids.add(model_id)
        logger.info(f"Model '{model_id}' loaded and pinned")

    async def warmup_inference(self) -> None:
        import numpy as np

        from speaches.audio import Audio
        from speaches.executors.shared.handler_protocol import SpeechRequest, TranscriptionRequest
        from speaches.executors.shared.vad_types import VadOptions

        tts_model_id: str | None = None
        tts_warmup_voice: str = "af_bella"
        for executor in self.text_to_speech:
            try:
                for model in executor.model_registry.list_local_models():
                    if model.id in self._pinned_model_ids:
                        tts_model_id = model.id
                        if hasattr(model, "voices") and model.voices:
                            tts_warmup_voice = model.voices[0].name
                        break
            except (OSError, ValueError):
                continue
            if tts_model_id:
                break

        stt_model_id: str | None = None
        for executor in self.transcription:
            try:
                for model in executor.model_registry.list_local_models():
                    if model.id in self._pinned_model_ids:
                        stt_model_id = model.id
                        break
            except (OSError, ValueError):
                continue
            if stt_model_id:
                break

        warmup_audio: Audio | None = None

        if tts_model_id:
            logger.info(f"Warmup inference: TTS with '{tts_model_id}'")
            tts_handler = self.resolve_tts_model_manager(tts_model_id)
            tts_request = SpeechRequest(model=tts_model_id, voice=tts_warmup_voice, text="warmup", speed=1.0)
            chunks = await asyncio.to_thread(lambda: list(tts_handler.handle_speech_request(tts_request)))
            if chunks:
                warmup_audio = Audio.concatenate(chunks)
            logger.info("Warmup inference: TTS complete")

        if stt_model_id:
            logger.info(f"Warmup inference: ASR with '{stt_model_id}'")
            stt_handler = self.resolve_stt_model_manager(stt_model_id)
            if warmup_audio is None:
                warmup_audio = Audio(np.zeros(16000, dtype=np.float32), sample_rate=16000)
            stt_request = TranscriptionRequest(
                audio=warmup_audio,
                model=stt_model_id,
                response_format="text",
                speech_segments=[],
                vad_options=VadOptions(),
                timestamp_granularities=["segment"],
            )
            await asyncio.to_thread(stt_handler.handle_non_streaming_transcription_request, stt_request)
            logger.info("Warmup inference: ASR complete")

    async def warmup_local_models(self) -> None:
        for executor in self.all_executors():
            try:
                local_models = list(executor.model_registry.list_local_models())
            except (OSError, ValueError):
                continue
            for model in local_models:
                if model.id in self._pinned_model_ids:
                    continue
                try:
                    disposable = await asyncio.to_thread(executor.model_manager.load_model, model.id)
                    loaded = self._exit_stack.enter_context(disposable)
                except (ImportError, RuntimeError):
                    logger.warning(f"Auto-warmup: skipping '{model.id}' (missing optional dependency)")
                    continue
                self._pinned_models.append(loaded)
                self._pinned_model_ids.add(model.id)
                logger.info(f"Auto-warmup: model '{model.id}' loaded and pinned")
