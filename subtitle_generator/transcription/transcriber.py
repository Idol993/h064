import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .model_loader import WhisperModelLoader


@dataclass
class TranscriptionSegment:
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    language: str = ""
    language_probability: float = 0.0


@dataclass
class TranscriptionResult:
    segments: list[TranscriptionSegment]
    language: str
    language_probability: float
    duration: float = 0.0


class AudioTranscriber:
    def __init__(
        self,
        model_loader: WhisperModelLoader,
        beam_size: int = 5,
        vad_filter: bool = True,
        temperature: float = 0.0
    ):
        self.model_loader = model_loader
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.temperature = temperature

    def transcribe(
        self,
        audio_path: str | os.PathLike,
        language: Optional[str] = None,
        progress_callback=None
    ) -> TranscriptionResult:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        model = self.model_loader.load()

        kwargs = {
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
        }

        if language is not None:
            kwargs["language"] = language

        if self.temperature > 0:
            kwargs["temperature"] = self.temperature

        segments_iter, info = model.transcribe(str(audio_path), **kwargs)

        segments: list[TranscriptionSegment] = []
        total_duration = info.duration if hasattr(info, "duration") else 0.0

        for idx, seg in enumerate(segments_iter, start=1):
            segment = TranscriptionSegment(
                id=seg.id if hasattr(seg, "id") else idx,
                start=round(seg.start, 2),
                end=round(seg.end, 2),
                text=seg.text.strip(),
                avg_logprob=getattr(seg, "avg_logprob", 0.0),
                language=info.language,
                language_probability=getattr(info, "language_probability", 0.0)
            )
            segments.append(segment)

            if progress_callback is not None and total_duration > 0:
                progress = min(seg.end / total_duration, 1.0)
                progress_callback(progress)

        if progress_callback is not None:
            progress_callback(1.0)

        return TranscriptionResult(
            segments=segments,
            language=info.language,
            language_probability=getattr(info, "language_probability", 0.0),
            duration=total_duration
        )
