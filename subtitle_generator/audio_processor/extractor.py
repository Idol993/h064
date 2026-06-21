import os
import tempfile
from pathlib import Path
from typing import Optional

import ffmpeg


class AudioExtractor:
    SUPPORTED_VIDEO_EXTENSIONS = {
        ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
        ".m4v", ".mpeg", ".mpg", ".3gp", ".ts", ".mts", ".m2ts"
    }
    SUPPORTED_AUDIO_EXTENSIONS = {
        ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
        ".opus", ".aiff", ".ape"
    }
    SUPPORTED_EXTENSIONS = SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels

    def is_supported_file(self, file_path: str | os.PathLike) -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS

    def is_video_file(self, file_path: str | os.PathLike) -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_VIDEO_EXTENSIONS

    def extract(
        self,
        input_path: str | os.PathLike,
        output_path: Optional[str | os.PathLike] = None
    ) -> str:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        if not self.is_supported_file(input_path):
            raise ValueError(f"不支持的文件格式: {input_path.suffix}")

        if output_path is None:
            temp_dir = Path(tempfile.gettempdir())
            output_path = temp_dir / f"{input_path.stem}_extracted.wav"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        ext = input_path.suffix.lower()

        if ext == ".wav":
            try:
                probe = ffmpeg.probe(str(input_path))
                audio_stream = next(
                    (s for s in probe["streams"] if s["codec_type"] == "audio"),
                    None
                )
                if audio_stream:
                    sr = int(audio_stream.get("sample_rate", 0))
                    ch = int(audio_stream.get("channels", 0))
                    if sr == self.sample_rate and ch == self.channels:
                        return str(input_path)
            except ffmpeg.Error:
                pass

        stream = ffmpeg.input(str(input_path))
        stream = ffmpeg.output(
            stream.audio,
            str(output_path),
            ac=self.channels,
            ar=self.sample_rate,
            format="wav",
            acodec="pcm_s16le",
            loglevel="error"
        )
        stream = ffmpeg.overwrite_output(stream)

        try:
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            raise RuntimeError(f"音频提取失败: {e.stderr.decode() if e.stderr else str(e)}")

        return str(output_path)
