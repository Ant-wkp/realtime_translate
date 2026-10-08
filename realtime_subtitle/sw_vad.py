from __future__ import annotations

import os

import numpy as np

from .compute import resolve_torch_device

SAMPLE_RATE = 16000
FRAME_SIZE = 512

class SoftwareVad:

    name = "abstract"

    def feed(self, audio: np.ndarray) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        pass

class SileroVad(SoftwareVad):

    name = "silero"

    def __init__(self, threshold: float = 0.5, device: str = "cpu"):
        self.threshold = threshold
        self.device = resolve_torch_device(device)
        self._carry = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0
        self._model = None
        self._load()

    def _load(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        jit_paths = [
            os.path.join(here, "silero_vad.jit"),
            "silero_vad.jit",
        ]
        try:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad()
            self._to_device()
            self.reset()
            return
        except ImportError:
            pass
        import torch

        for path in jit_paths:
            if os.path.isfile(path):
                self._model = torch.jit.load(path)
                self._to_device()
                self.reset()
                return
        raise RuntimeError(
            "Silero VAD requires the silero-vad package or a local "
            "silero_vad.jit model file"
        )

    def _to_device(self) -> None:
        move = getattr(self._model, "to", None)
        if move is None:
            return
        try:
            self._model = move(self.device)
        except Exception:  # noqa: BLE001 - keep the model where it loaded
            pass

    def feed(self, audio: np.ndarray) -> float:
        if audio is None or audio.size == 0:
            return self._last_prob

        block = np.concatenate([self._carry, audio.astype(np.float32)])
        n_frames = block.size // FRAME_SIZE
        if n_frames == 0:
            self._carry = block
            return self._last_prob

        frames = block[: n_frames * FRAME_SIZE].reshape(n_frames, FRAME_SIZE)
        self._carry = block[n_frames * FRAME_SIZE :]

        for frame in frames:
            self._last_prob = self._infer_frame(frame)
        return self._last_prob

    def _infer_frame(self, frame: np.ndarray) -> float:
        import torch

        with torch.no_grad():
            audio = torch.from_numpy(frame.astype(np.float32)).unsqueeze(0)
            if self.device != "cpu":
                audio = audio.to(self.device)
            prob = self._model(
                audio,
                torch.tensor(SAMPLE_RATE, device=audio.device),
            )
        return float(prob.item())

    def reset(self) -> None:
        self._carry = np.zeros(0, dtype=np.float32)
        if self._model is not None:
            self._model.reset_states()

    @property
    def last_probability(self) -> float:
        return self._last_prob
