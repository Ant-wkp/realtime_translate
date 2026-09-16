"""Software VAD gating layer (Silero) cascaded with the hardware VAD.

Silero VAD v5 runs ONNX-first with JIT fallback, frames audio in fixed
512-sample chunks with a carry buffer so block sizes from the audio callback
need not align, and only resets internal RNN state when an utterance has
truly ended (never on segment start, which would cold-start the model).
"""

from __future__ import annotations

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
    """Silero VAD v5 wrapper: ONNX runtime first, torch JIT fallback.

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
        self._backend = None
        self._ctx = None  # ONNX inference session state or torch RNN state
        self._load()

    # ------------------------------------------------------------------
    # model loading
    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            import onnxruntime as ort

            self._model = ort.InferenceSession(
                "silero_vad.onnx", providers=["CPUExecutionProvider"]
            )
            self._backend = "onnx"
            self.reset()
            return
        except Exception:  # noqa: BLE001 - fall through to torch
            pass
        try:
            import torch

            self._torch = torch
            self._model = torch.jit.load("silero_vad.jit")
            self._backend = "jit"
            self.reset()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Silero VAD requires onnxruntime (silero_vad.onnx) or torch "
                "(silero_vad.jit); neither could be loaded"
            ) from exc

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
        if self._backend == "onnx":
            import numpy as _np

            inp = _np.expand_dims(frame, axis=0).astype(_np.float32)
            sr = _np.array([SAMPLE_RATE], dtype=_np.int64)
            outs = self._model.run(None, {"input": inp, "sr": sr, "state": self._ctx})
            prob, self._ctx = outs[0], outs[2]
            return float(prob[0][0])

        torch = self._torch
        with torch.no_grad():
            prob, self._ctx, _ = self._model(
                torch.from_numpy(frame).unsqueeze(0), SAMPLE_RATE, self._ctx
            )
        return float(prob.item())

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset RNN state. Only call after an utterance truly ended."""
        self._carry = np.zeros(0, dtype=np.float32)
        if self._backend == "onnx":
            import numpy as _np

            self._ctx = _np.zeros((2, 1, 128), dtype=_np.float32)
        else:
            self._ctx = None
            self._model.reset_state()

    @property
    def last_probability(self) -> float:
        return self._last_prob

    def is_speech(self, audio: np.ndarray) -> bool:
        return self.feed(audio) >= self.threshold
