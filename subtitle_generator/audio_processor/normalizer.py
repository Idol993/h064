import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False


class AudioNormalizer:
    def __init__(self, target_lufs: float = -16.0):
        self.target_lufs = target_lufs

    def normalize(
        self,
        input_path: str | os.PathLike,
        output_path: Optional[str | os.PathLike] = None
    ) -> str:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_normalized.wav"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        audio, sample_rate = sf.read(str(input_path), always_2d=True)
        audio_mono = audio[:, 0] if audio.ndim > 1 else audio

        if HAS_PYLOUDNORM:
            normalized = self._normalize_loudnorm(audio_mono, sample_rate)
        else:
            normalized = self._normalize_peak(audio_mono)

        if audio.ndim > 1:
            normalized = np.column_stack([normalized] * audio.shape[1])

        sf.write(str(output_path), normalized, sample_rate, subtype="PCM_16")
        return str(output_path)

    def _normalize_loudnorm(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(audio)

        if np.isinf(loudness) or np.isnan(loudness):
            return self._normalize_peak(audio)

        try:
            normalized = pyln.normalize.loudness(audio, loudness, self.target_lufs)
            peak = np.max(np.abs(normalized))
            if peak > 1.0:
                normalized = normalized / peak * 0.99
            return normalized
        except Exception:
            return self._normalize_peak(audio)

    def _normalize_peak(self, audio: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(audio))
        if peak == 0:
            return audio
        target_peak = 0.95
        return audio * (target_peak / peak)
