import os
from datetime import timedelta
from pathlib import Path

import srt


class SRTFormatter:
    def __init__(self):
        pass

    @staticmethod
    def _format_timestamp(seconds: float) -> timedelta:
        return timedelta(seconds=seconds)

    def format(
        self,
        segments: list,
        output_path: str | os.PathLike
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        srt_subtitles = []
        for idx, seg in enumerate(segments, start=1):
            subtitle = srt.Subtitle(
                index=idx,
                start=self._format_timestamp(seg.start),
                end=self._format_timestamp(seg.end),
                content=seg.text
            )
            srt_subtitles.append(subtitle)

        srt_content = srt.compose(srt_subtitles)
        output_path.write_text(srt_content, encoding="utf-8")

        return str(output_path)
