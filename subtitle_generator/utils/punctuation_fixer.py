import re


class PunctuationFixer:
    EN_PUNCT = set('.,!?;:')
    ZH_PUNCT_MAP = {
        '.': '。',
        ',': '，',
        '!': '！',
        '?': '？',
        ';': '；',
        ':': '：',
    }
    EN_PUNCT_MAP = {v: k for k, v in ZH_PUNCT_MAP.items()}

    def __init__(self):
        pass

    @staticmethod
    def _has_chinese(text: str) -> bool:
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    @staticmethod
    def _has_english(text: str) -> bool:
        return bool(re.search(r'[a-zA-Z]', text))

    def fix(self, text: str) -> str:
        if not text:
            return text

        text = self._fix_spacing(text)
        text = self._fix_punctuation_context(text)
        text = self._fix_repeated_punctuation(text)
        text = self._fix_mixed_quotes(text)

        return text

    def _fix_spacing(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)
        text = re.sub(r'([a-zA-Z0-9])([\u4e00-\u9fff])', r'\1 \2', text)
        text = re.sub(r'([\u4e00-\u9fff])([a-zA-Z0-9])', r'\1 \2', text)
        return text.strip()

    def _fix_punctuation_context(self, text: str) -> str:
        chars = list(text)
        result = []

        for i, ch in enumerate(chars):
            prev_char = chars[i - 1] if i > 0 else ''
            next_char = chars[i + 1] if i < len(chars) - 1 else ''

            if ch in self.ZH_PUNCT_MAP:
                prev_has_zh = self._has_chinese(prev_char)
                prev_has_en = self._has_english(prev_char)
                next_has_zh = self._has_chinese(next_char)

                if (prev_has_zh or next_has_zh) and not prev_has_en:
                    result.append(self.ZH_PUNCT_MAP[ch])
                else:
                    result.append(ch)
            elif ch in self.EN_PUNCT_MAP:
                prev_has_en = self._has_english(prev_char)
                prev_has_zh = self._has_chinese(prev_char)
                next_has_en = self._has_english(next_char)

                if prev_has_en and (next_has_en or not prev_has_zh):
                    result.append(self.EN_PUNCT_MAP[ch])
                else:
                    result.append(ch)
            else:
                result.append(ch)

        return ''.join(result)

    def _fix_repeated_punctuation(self, text: str) -> str:
        text = re.sub(r'([。！？!?，,；;、:：])\1+', r'\1', text)
        text = re.sub(r'([。！？！？]){2,}', r'\1\1', text)
        return text

    def _fix_mixed_quotes(self, text: str) -> str:
        text = re.sub(r'"([^"]*)"', lambda m: self._quote_for_context(m.group(1)), text)
        text = re.sub(r"'([^']*)'", lambda m: self._single_quote_for_context(m.group(1)), text)
        return text

    def _quote_for_context(self, content: str) -> str:
        if self._has_chinese(content):
            return f'"{content}"'
        return f'"{content}"'

    def _single_quote_for_context(self, content: str) -> str:
        if self._has_chinese(content):
            return f"'{content}'"
        return f"'{content}'"
