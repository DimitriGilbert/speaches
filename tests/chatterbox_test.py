"""Tests for the chatterbox executor and the clone-voices endpoint.

The real ``chatterbox-tts`` package is an optional dependency (the ``chatterbox``
extra) and pulls in heavy model weights. These tests inject a fake
``chatterbox.tts.ChatterboxTTS`` into ``sys.modules`` so the executor module
believes the library is importable, which lets ``ChatterboxModelManager`` be
defined and exercised without the real model.
"""

from __future__ import annotations

import io
import pathlib
import shutil
import sys
import types
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

if TYPE_CHECKING:
    from collections.abc import Generator

# Reuse the route's wav-header check in the transcode test.
from speaches.routers.voices import _is_wav as _is_wav_imported

CHATTERBOX_MODEL_ID = "ResembleAI/chatterbox"
EXPECTED_SAMPLE_RATE = 24000


class _FakeTensor:
    """Mimics the small slice of the torch tensor API the executor uses."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _FakeChatterboxTTS:
    """Stand-in for ``chatterbox.tts.ChatterboxTTS`` used by the executor."""

    last_generate_kwargs: dict | None = None

    # The real ChatterboxTTS exposes the sample rate as `sr` (generate() returns
    # only the audio tensor, not a (audio, sr) tuple).
    sr = EXPECTED_SAMPLE_RATE

    @classmethod
    def from_pretrained(cls, device: str = "cpu") -> _FakeChatterboxTTS:  # noqa: ARG003
        return cls()

    def generate(self, text: str, audio_prompt_path: str | None = None) -> _FakeTensor:
        type(self).last_generate_kwargs = {"text": text, "audio_prompt_path": audio_prompt_path}
        # Real Chatterbox.generate() returns a torch.Tensor of shape (1, N)
        # (mono as 2-D). Mirror that so the executor's flatten path is exercised.
        return _FakeTensor(np.zeros((1, 100), dtype=np.float32))


@pytest.fixture(autouse=True)
def _reset_fake_generate_call() -> Generator[None]:
    """Clear the recorded ``generate`` kwargs between tests."""
    _FakeChatterboxTTS.last_generate_kwargs = None
    yield
    _FakeChatterboxTTS.last_generate_kwargs = None


@pytest.fixture
def chatterbox_executor() -> Generator[tuple[types.ModuleType, type]]:
    """Import the executor with a fake ``chatterbox.tts`` available in ``sys.modules``.

    The executor module is reloaded (after injecting the fake modules) so its
    top-level ``try: from chatterbox.tts import ChatterboxTTS`` succeeds, making
    ``CHATTERBOX_AVAILABLE`` True and ``ChatterboxModelManager`` defined. The real
    modules and the previously imported executor module are restored on teardown.
    """
    # ``ModuleType`` is what ``sys.modules`` expects; attribute assignment on it
    # is normal at runtime but pyrefly can't see the synthetic attributes.
    fake_pkg = types.ModuleType("chatterbox")
    fake_tts = types.ModuleType("chatterbox.tts")
    fake_tts.ChatterboxTTS = _FakeChatterboxTTS  # type: ignore[missing-attribute]
    fake_pkg.tts = fake_tts  # type: ignore[missing-attribute]

    saved_modules = {
        "chatterbox": sys.modules.get("chatterbox"),
        "chatterbox.tts": sys.modules.get("chatterbox.tts"),
        "speaches.executors.chatterbox": sys.modules.get("speaches.executors.chatterbox"),
        "speaches.executors.shared.registry": sys.modules.get("speaches.executors.shared.registry"),
    }
    sys.modules["chatterbox"] = fake_pkg
    sys.modules["chatterbox.tts"] = fake_tts
    sys.modules.pop("speaches.executors.chatterbox", None)
    # The registry imports chatterbox symbols at module load; drop its cached
    # import so it re-evaluates CHATTERBOX_AVAILABLE against the fake module.
    sys.modules.pop("speaches.executors.shared.registry", None)

    import speaches.executors.chatterbox as chatterbox_mod

    assert chatterbox_mod.CHATTERBOX_AVAILABLE is True, "fake chatterbox.tts was not picked up"
    manager_cls = getattr(chatterbox_mod, "ChatterboxModelManager", None)
    assert manager_cls is not None, "ChatterboxModelManager is not defined despite CHATTERBOX_AVAILABLE=True"

    yield chatterbox_mod, manager_cls

    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


# --- Test 1: registration gating -------------------------------------------------


def test_chatterbox_available_flag_when_import_absent() -> None:
    """Without the ``chatterbox`` extra installed the executor must self-disable.

    The local dev environment intentionally lacks ``chatterbox-tts`` (it is an
    optional extra), so the module-level gate evaluates to False and the
    ``ChatterboxModelManager`` class is never defined. The registry instance is
    always defined regardless, and ``ExecutorRegistry`` must construct cleanly
    without a chatterbox executor.
    """
    from speaches.config import Config
    from speaches.executors.chatterbox import CHATTERBOX_AVAILABLE, chatterbox_model_registry
    from speaches.executors.shared.registry import ExecutorRegistry

    assert CHATTERBOX_AVAILABLE is False
    # The registry instance exists even when the library is unavailable.
    assert chatterbox_model_registry is not None
    # And the registry does not register a chatterbox executor when unavailable.
    config = Config(tts_model_ttl=-1, enable_ui=False)
    registry = ExecutorRegistry(config)
    tts_names = [executor.name for executor in registry.text_to_speech]
    assert "chatterbox" not in tts_names


# --- Test 2: preset generation (mocked model) ------------------------------------


def test_handle_speech_request_preset(
    chatterbox_executor: tuple[types.ModuleType, type],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """A preset generation call yields one float32 chunk at the chatterbox sample rate."""
    chatterbox_mod, manager_cls = chatterbox_executor

    # Declare "default" as a valid preset voice and ensure no clone file matches
    # by pointing CLONE_VOICES_DIR at an empty tmp dir.
    monkeypatch.setattr(chatterbox_mod, "PREDEFINED_VOICE_NAMES", ["default"])
    monkeypatch.setattr(chatterbox_mod, "CLONE_VOICES_DIR", tmp_path)
    manager = manager_cls(ttl=-1)

    from speaches.executors.shared.handler_protocol import SpeechRequest

    request = SpeechRequest(model=CHATTERBOX_MODEL_ID, voice="default", text="hello", speed=1.0)
    chunks = list(manager.handle_speech_request(request))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.sample_rate == EXPECTED_SAMPLE_RATE
    assert chunk.data.dtype == np.float32
    assert chunk.data.shape == (100,)
    # generate() must have been called WITHOUT an audio prompt for a preset voice.
    assert _FakeChatterboxTTS.last_generate_kwargs is not None
    assert _FakeChatterboxTTS.last_generate_kwargs["audio_prompt_path"] is None


# --- Test 3: clone generation (mocked model) -------------------------------------


def test_handle_speech_request_clone_passes_audio_prompt(
    chatterbox_executor: tuple[types.ModuleType, type],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """When a clone ``.wav`` exists its path is forwarded to ``generate``."""
    chatterbox_mod, manager_cls = chatterbox_executor

    clone_dir = tmp_path / "voices"
    clone_dir.mkdir()
    stem = "cloned-voice"
    clone_file = clone_dir / f"{stem}.wav"
    # Write a real (tiny) wav so ``Path.exists()`` is truthful.
    sf.write(clone_file, np.zeros(10, dtype=np.float32), EXPECTED_SAMPLE_RATE, format="WAV")

    monkeypatch.setattr(chatterbox_mod, "CLONE_VOICES_DIR", clone_dir)

    manager = manager_cls(ttl=-1)

    from speaches.executors.shared.handler_protocol import SpeechRequest

    request = SpeechRequest(model=CHATTERBOX_MODEL_ID, voice=stem, text="hello", speed=1.0)
    chunks = list(manager.handle_speech_request(request))

    assert len(chunks) == 1
    assert chunks[0].sample_rate == EXPECTED_SAMPLE_RATE
    assert _FakeChatterboxTTS.last_generate_kwargs is not None
    assert _FakeChatterboxTTS.last_generate_kwargs["audio_prompt_path"] == str(clone_file)


def test_handle_speech_request_unknown_voice_raises(
    chatterbox_executor: tuple[types.ModuleType, type],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """A voice that is neither preset nor an existing clone file is rejected."""
    chatterbox_mod, manager_cls = chatterbox_executor
    # No presets, and point the clone dir at an empty tmp dir.
    monkeypatch.setattr(chatterbox_mod, "PREDEFINED_VOICE_NAMES", [])
    monkeypatch.setattr(chatterbox_mod, "CLONE_VOICES_DIR", tmp_path)

    manager = manager_cls(ttl=-1)

    from speaches.executors.shared.handler_protocol import SpeechRequest

    request = SpeechRequest(model=CHATTERBOX_MODEL_ID, voice="no-such-voice", text="hello", speed=1.0)
    with pytest.raises(ValueError, match="not supported"):
        list(manager.handle_speech_request(request))


# --- Test 4: clone surfacing in list_local_models --------------------------------


def test_list_local_models_attaches_clones_as_voices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Clone ``.wav`` files attach to the chatterbox model as voices.

    Clones must NOT surface as separate model entries: the UI derives the voice
    dropdown from the selected model's voices[], so a clone under its own id
    would leave the real chatterbox model with an empty voice list.
    """
    import speaches.executors.chatterbox as chatterbox_mod

    clone_dir = tmp_path / "voices"
    clone_dir.mkdir()
    for stem in ("foo", "bar"):
        sf.write(clone_dir / f"{stem}.wav", np.zeros(10, dtype=np.float32), EXPECTED_SAMPLE_RATE, format="WAV")

    monkeypatch.setattr(chatterbox_mod, "CLONE_VOICES_DIR", clone_dir)

    # Stub the HF cache scan to return exactly one cached chatterbox repo, so
    # the test does not depend on a real download. The repo must pass the
    # chatterbox library_name filter to be yielded by the first loop.
    class _FakeRepoInfo:
        repo_id = CHATTERBOX_MODEL_ID
        last_modified = 0.0

    class _FakeCardData:
        library_name = "chatterbox"
        tags = None

    fake_repos = [_FakeRepoInfo()]

    def _fake_get_card_data(_repo_info: object) -> _FakeCardData:
        return _FakeCardData()

    monkeypatch.setattr(chatterbox_mod, "get_cached_model_repos_info", lambda: fake_repos)
    monkeypatch.setattr(chatterbox_mod, "get_model_card_data_from_cached_repo_info", _fake_get_card_data)
    # list_remote_models calls extract_language_list on the card; not needed
    # here, but keep the attribute present to avoid attribute errors elsewhere.
    monkeypatch.setattr(chatterbox_mod, "extract_language_list", lambda _card: ["en"])

    models = list(chatterbox_mod.chatterbox_model_registry.list_local_models())
    # Exactly one model (the chatterbox repo); clones are NOT separate models.
    assert [m.id for m in models] == [CHATTERBOX_MODEL_ID]
    voice_names = [v.name for v in models[0].voices]
    assert "foo" in voice_names
    assert "bar" in voice_names


# --- Test 5: regression — other TTS executors unaffected ------------------------


def test_regression_other_tts_unchanged() -> None:
    """Chatterbox being absent must not drop the kokoro/piper TTS executors."""
    from speaches.config import Config
    from speaches.executors.shared.registry import ExecutorRegistry

    config = Config(tts_model_ttl=-1, enable_ui=False)
    registry = ExecutorRegistry(config)
    tts_names = [executor.name for executor in registry.text_to_speech]
    assert "chatterbox" not in tts_names
    assert "kokoro" in tts_names
    assert "piper" in tts_names


# --- Test 6: clone upload endpoint ----------------------------------------------


def _tiny_wav_bytes() -> bytes:
    """Synthesize a minimal valid WAV blob (44-byte header + a few zero samples)."""
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(50, dtype=np.float32), EXPECTED_SAMPLE_RATE, format="WAV")
    return buffer.getvalue()


@pytest.fixture
def voices_client(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> TestClient:
    """A TestClient wired to a minimal app exposing only the voices router.

    ``CLONE_VOICES_DIR`` is redirected to a per-test tmp dir so the tests never
    touch the real user cache.
    """
    import speaches.routers.voices as voices_router_mod

    monkeypatch.setattr(voices_router_mod, "CLONE_VOICES_DIR", tmp_path)
    app = FastAPI()
    app.include_router(voices_router_mod.router)
    return TestClient(app)


def test_clone_upload_endpoint_creates_voice(voices_client: TestClient) -> None:
    response = voices_client.post(
        "/v1/audio/voices",
        data={"name": "my-voice"},
        files={"file": ("my-voice.wav", _tiny_wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["id"] == "my-voice"
    assert body["name"] == "my-voice"
    assert pathlib.Path(body["path"]).name == "my-voice.wav"


def test_clone_upload_endpoint_rejects_invalid_name(voices_client: TestClient) -> None:
    # Path-traversal-style and uppercase names must be rejected (400).
    for bad_name in ("../etc", "UPPER CASE", "with/slash"):
        response = voices_client.post(
            "/v1/audio/voices",
            data={"name": bad_name},
            files={"file": ("x.wav", _tiny_wav_bytes(), "audio/wav")},
        )
        assert response.status_code == 400, f"expected 400 for name={bad_name!r}, got {response.status_code}"


def test_clone_upload_endpoint_conflicts_on_duplicate(voices_client: TestClient) -> None:
    wav_bytes = _tiny_wav_bytes()
    first = voices_client.post(
        "/v1/audio/voices",
        data={"name": "dup"},
        files={"file": ("dup.wav", wav_bytes, "audio/wav")},
    )
    assert first.status_code == 201
    second = voices_client.post(
        "/v1/audio/voices",
        data={"name": "dup"},
        files={"file": ("dup.wav", wav_bytes, "audio/wav")},
    )
    assert second.status_code == 409


def test_clone_upload_endpoint_transcodes_non_wav(
    voices_client: TestClient,
    tmp_path: pathlib.Path,
) -> None:
    # Mic recordings arrive as non-wav (webm/opus, mp3, ...). The endpoint must
    # transcode them to wav via ffmpeg rather than reject them. Generate a tiny
    # FLAC blob (soundfile supports it without ffmpeg) as the "non-wav" upload.
    flac_buffer = io.BytesIO()
    sf.write(flac_buffer, np.zeros(50, dtype=np.float32), EXPECTED_SAMPLE_RATE, format="FLAC")
    flac_bytes = flac_buffer.getvalue()
    assert not _is_wav_imported(flac_bytes)  # sanity: it really isn't wav

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed; transcode path cannot be exercised")

    response = voices_client.post(
        "/v1/audio/voices",
        data={"name": "from-mic"},
        files={"file": ("from-mic.flac", flac_bytes, "audio/flac")},
    )
    assert response.status_code == 201
    saved = tmp_path / "from-mic.wav"
    assert saved.exists()
    # The transcoded file must be a real wav (RIFF/WAVE header), not the raw flac.
    assert _is_wav_imported(saved.read_bytes())
