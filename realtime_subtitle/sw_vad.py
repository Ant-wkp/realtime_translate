"""Software VAD gating layer (Silero) cascaded with the hardware VAD.

Silero VAD runs on the torch JIT model (official silero-vad package
preferred, local silero_vad.jit fallback), frames audio in fixed
512-sample chunks with a carry buffer so block sizes from the audio callback
need not align, and only resets internal RNN state when an utterance has
truly ended (never on segment start, which would cold-start the model).
The v5 ONNX export is intentionally not used: it misreports speech
probability (~0.001 on real speech).
"""

from __future__ import annotations

import os

import numpy as np

SAMPLE_RATE = 16000
FRAME_SIZE = 512  # 32 ms at 16 kHz


class SoftwareVad:
    """Abstract software VAD interface."""

    name = "abstract"

    def feed(self, audio: np.ndarray) -> float:
        """Consume one audio block, return latest speech probability [0, 1]."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal state after an utterance truly ended."""
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        pass


class SileroVad(SoftwareVad):
    """Silero VAD wrapper on the torch JIT backend.

    Frames are always 512 samples (32 ms @ 16 kHz). Leftover samples are kept
    in a carry buffer and prepended to the next block so callers can push
    arbitrarily sized float32 blocks.
    """

    name = "silero"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._carry = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0
        self._model = None
        self._load()

    # ------------------------------------------------------------------
    # model loading
    # ------------------------------------------------------------------
    def _load(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        # script mode (python main.py) runs with CWD = package dir, module
        # mode runs from the repo root - try both, plus plain CWD.
        jit_paths = [
            os.path.join(here, "silero_vad.jit"),
            "silero_vad.jit",
        ]
        # 1) official pip package (silero-vad) - preferred, ships the
        #    correct torch jit model
        try:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad()
            self.reset()
            return
        except ImportError:
            pass
        # 2) local torch jit file
        import torch

        for path in jit_paths:
            if os.path.isfile(path):
                self._model = torch.jit.load(path)
                self.reset()
                return
        raise RuntimeError(
            "Silero VAD requires the silero-vad package or a local "
            "silero_vad.jit model file"
        )

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
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
        # jit backend (official package or local file): m(x, sr) -> prob
        # the model keeps its own hidden state, no external ctx needed
        import torch

        with torch.no_grad():
            prob = self._model(
                torch.from_numpy(frame.astype(np.float32)).unsqueeze(0),
                torch.tensor(SAMPLE_RATE),
            )
        return float(prob.item())

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset RNN state. Only call after an utterance truly ended."""
        self._carry = np.zeros(0, dtype=np.float32)
        if self._model is not None:
            self._model.reset_states()

    @property
    def last_probability(self) -> float:
        return self._last_prob

    def is_speech(self, audio: np.ndarray) -> bool:
        return self.feed(audio) >= self.threshold
