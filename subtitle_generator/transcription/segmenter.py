import re
from dataclasses import dataclass


@dataclass
class SegmentChunk:
    start: float
    end: float
    text: str


class TextSegmenter:
    def __init__(
        self,
        max_chinese_chars: int = 50,
        max_english_chars: int = 80,
        min_duration: float = 0.5,
        max_duration: float = 8.0
    ):
        if max_chinese_chars <= 0:
            raise ValueError(f"max_chinese_chars 必须为正整数，当前值: {max_chinese_chars}")
        if max_english_chars <= 0:
            raise ValueError(f"max_english_chars 必须为正整数，当前值: {max_english_chars}")
        if min_duration <= 0:
            raise ValueError(f"min_duration 必须大于 0，当前值: {min_duration}")
        if max_duration <= 0:
            raise ValueError(f"max_duration 必须大于 0，当前值: {max_duration}")
        if min_duration >= max_duration:
            raise ValueError(f"min_duration ({min_duration}) 必须小于 max_duration ({max_duration})")

        self.max_chinese_chars = max_chinese_chars
        self.max_english_chars = max_english_chars
        self.min_duration = min_duration
        self.max_duration = max_duration

    @staticmethod
    def _count_chinese(text: str) -> int:
        return len(re.findall(r"[\u4e00-\u9fff]", text))

    @staticmethod
    def _count_english(text: str) -> int:
        return len(re.findall(r"[a-zA-Z]", text))

    def _is_too_long(self, text: str) -> bool:
        chinese_chars = self._count_chinese(text)
        english_chars = self._count_english(text)
        total = chinese_chars + english_chars

        if chinese_chars > english_chars:
            return len(text) > self.max_chinese_chars
        else:
            return total > self.max_english_chars

    def _split_at_punctuation(self, text: str) -> list[str]:
        pattern = r'([。！？!?，,；;、])'
        parts = re.split(pattern, text)

        chunks: list[str] = []
        current = ""

        for i in range(0, len(parts), 2):
            part = parts[i]
            punct = parts[i + 1] if i + 1 < len(parts) else ""

            if not part and not punct:
                continue

            candidate = current + part + punct

            if self._is_too_long(candidate) and current:
                chunks.append(current.strip())
                current = part + punct
            else:
                current = candidate

        if current.strip():
            chunks.append(current.strip())

        if len(chunks) == 1 and self._is_too_long(chunks[0]):
            chunks = self._split_by_length(chunks[0])

        return [c for c in chunks if c]

    def _split_by_length(self, text: str) -> list[str]:
        chunks: list[str] = []
        chunk_size = self.max_chinese_chars if self._count_chinese(text) > self._count_english(text) else self.max_english_chars

        while text:
            if len(text) <= chunk_size:
                chunks.append(text.strip())
                break

            split_pos = chunk_size
            for i in range(chunk_size, max(0, chunk_size - 20), -1):
                if text[i - 1] in " ，,。.！!？?；;、":
                    split_pos = i
                    break

            chunks.append(text[:split_pos].strip())
            text = text[split_pos:].strip()

        return [c for c in chunks if c]

    def segment(
        self,
        segments: list
    ) -> list[SegmentChunk]:
        result: list[SegmentChunk] = []

        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue

            duration = seg.end - seg.start

            if not self._is_too_long(text) and duration <= self.max_duration:
                result.append(SegmentChunk(
                    start=seg.start,
                    end=seg.end,
                    text=text
                ))
                continue

            if duration <= self.min_duration:
                result.append(SegmentChunk(
                    start=seg.start,
                    end=seg.end,
                    text=text
                ))
                continue

            text_chunks = self._split_at_punctuation(text)

            if len(text_chunks) == 1:
                result.append(SegmentChunk(
                    start=seg.start,
                    end=seg.end,
                    text=text
                ))
                continue

            chars_per_second = len(text) / max(duration, 0.01)
            time_cursor = seg.start

            for i, chunk in enumerate(text_chunks):
                chunk_len = len(chunk)
                chunk_duration = chunk_len / chars_per_second
                chunk_duration = max(chunk_duration, self.min_duration)

                if i == len(text_chunks) - 1:
                    chunk_end = seg.end
                else:
                    chunk_end = min(time_cursor + chunk_duration, seg.end)

                result.append(SegmentChunk(
                    start=round(time_cursor, 2),
                    end=round(chunk_end, 2),
                    text=chunk
                ))

                time_cursor = chunk_end

        return result
