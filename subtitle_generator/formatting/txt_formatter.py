import os
from pathlib import Path


class TXTFormatter:
    def __init__(self, with_timestamp: bool = False):
        self.with_timestamp = with_timestamp

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"

    def format(
        self,
        segments: list,
        output_path: str | os.PathLike
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for seg in segments:
            if self.with_timestamp:
                timestamp = self._format_timestamp(seg.start)
                lines.append(f"{timestamp} {seg.text}")
            else:
                lines.append(seg.text)

        txt_content = "\n".join(lines) + "\n"
        output_path.write_text(txt_content, encoding="utf-8")

        return str(output_path)
