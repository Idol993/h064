import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


class AudioDenoiser:
    def __init__(self, level: float = 0.5):
        if not 0.0 <= level <= 1.0:
            raise ValueError("降噪强度必须在 0.0 到 1.0 之间")
        self.level = level

    def denoise(
        self,
        input_path: str | os.PathLike,
        output_path: Optional[str | os.PathLike] = None
    ) -> str:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_denoised.wav"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        audio, sample_rate = sf.read(str(input_path), always_2d=True)
        audio = audio[:, 0] if audio.ndim > 1 else audio

        denoised = self._spectral_subtraction(audio, sample_rate)

        sf.write(str(output_path), denoised, sample_rate, subtype="PCM_16")
        return str(output_path)

    def _spectral_subtraction(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.level == 0.0:
            return audio

        frame_size = int(sample_rate * 0.025)
        hop_size = int(sample_rate * 0.01)

        window = np.hanning(frame_size)
        n_frames = 1 + (len(audio) - frame_size) // hop_size

        if n_frames < 5:
            return audio

        noise_frames = min(10, n_frames // 10)
        noise_spec_sum = np.zeros(frame_size)

        for i in range(noise_frames):
            start = i * hop_size
            frame = audio[start:start + frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))
            spectrum = np.abs(np.fft.fft(frame * window))
            noise_spec_sum += spectrum

        noise_spec = noise_spec_sum / noise_frames

        output = np.zeros_like(audio)

        for i in range(n_frames):
            start = i * hop_size
            end = start + frame_size
            frame = audio[start:end]

            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))

            fft_result = np.fft.fft(frame * window)
            magnitude = np.abs(fft_result)
            phase = np.angle(fft_result)

            alpha = 2.0 - self.level * 1.0
            beta = 0.01 + self.level * 0.02

            subtraction = magnitude ** alpha - self.level * noise_spec ** alpha
            subtraction = np.maximum(subtraction, beta * noise_spec ** alpha)
            cleaned_mag = subtraction ** (1.0 / alpha)

            cleaned_fft = cleaned_mag * np.exp(1j * phase)
            cleaned_frame = np.fft.ifft(cleaned_fft).real
            cleaned_frame = cleaned_frame * window

            output[start:end] += cleaned_frame

        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.95

        return output
