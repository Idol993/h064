import os
from pathlib import Path


class VTTFormatter:
    def __init__(self):
        pass

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        total_ms = int(seconds * 1000)
        hours = total_ms // 3600000
        minutes = (total_ms % 3600000) // 60000
        secs = (total_ms % 60000) // 1000
        ms = total_ms % 1000
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
        else:
            return f"{minutes:02d}:{secs:02d}.{ms:03d}"

    def format(
        self,
        segments: list,
        output_path: str | os.PathLike
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = ["WEBVTT", ""]

        for seg in segments:
            start_ts = self._format_timestamp(seg.start)
            end_ts = self._format_timestamp(seg.end)
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(seg.text)
            lines.append("")

        vtt_content = "\n".join(lines).rstrip() + "\n"
        output_path.write_text(vtt_content, encoding="utf-8")

        return str(output_path)
