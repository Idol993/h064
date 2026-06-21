import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TxtSegment:
    start: float
    end: float
    text: str


class TXTFormatter:
    TIMESTAMP_PATTERN = re.compile(
        r'^\s*\[?\s*(\d{1,2}):(\d{2}):(\d{2})(?:[.,](\d{1,3}))?\s*\]?\s*(.*)$'
    )
    SIMPLE_TIMESTAMP_PATTERN = re.compile(
        r'^\s*\[?\s*(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?\s*\]?\s*(.*)$'
    )

    def __init__(self, with_timestamp: bool = False):
        self.with_timestamp = with_timestamp

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"

    @classmethod
    def _parse_timestamp_line(cls, line: str) -> tuple[float | None, str]:
        m = cls.TIMESTAMP_PATTERN.match(line)
        if m:
            h, mi, s, ms, text = m.groups()
            total = int(h) * 3600 + int(mi) * 60 + int(s)
            if ms:
                total += int(ms.ljust(3, "0")[:3]) / 1000.0
            return total, text.strip()

        m2 = cls.SIMPLE_TIMESTAMP_PATTERN.match(line)
        if m2:
            mi, s, ms, text = m2.groups()
            total = int(mi) * 60 + int(s)
            if ms:
                total += int(ms.ljust(3, "0")[:3]) / 1000.0
            return total, text.strip()

        return None, line.strip()

    @classmethod
    def parse(cls, file_path: str | os.PathLike, default_duration: float = 3.0) -> list[TxtSegment]:
        file_path = Path(file_path)
        lines = file_path.read_text(encoding="utf-8").splitlines()

        segments: list[TxtSegment] = []

        timed_entries: list[tuple[float, str]] = []
        plain_lines: list[str] = []
        has_timestamps = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            ts, text = cls._parse_timestamp_line(stripped)
            if ts is not None and text:
                has_timestamps = True
                timed_entries.append((ts, text))
            elif ts is not None and not text:
                has_timestamps = True
            else:
                if has_timestamps and timed_entries:
                    timed_entries[-1] = (timed_entries[-1][0], timed_entries[-1][1] + " " + stripped)
                else:
                    plain_lines.append(stripped)

        if has_timestamps and timed_entries:
            for i, (start, text) in enumerate(timed_entries):
                if i + 1 < len(timed_entries):
                    end = timed_entries[i + 1][0]
                else:
                    end = start + default_duration
                if end <= start:
                    end = start + default_duration
                segments.append(TxtSegment(start=round(start, 3), end=round(end, 3), text=text))
        else:
            for i, text in enumerate(plain_lines):
                start = float(i) * default_duration
                end = start + default_duration * 0.85
                segments.append(TxtSegment(start=round(start, 3), end=round(end, 3), text=text))

        return segments

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
