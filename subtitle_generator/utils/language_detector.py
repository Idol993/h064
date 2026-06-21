LANGUAGE_NAMES = {
    "zh": "中文",
    "zh-CN": "简体中文",
    "zh-TW": "繁体中文",
    "en": "English",
    "ja": "日本語",
    "ko": "한국어",
    "fr": "Français",
    "de": "Deutsch",
    "es": "Español",
    "ru": "Русский",
    "it": "Italiano",
    "pt": "Português",
    "ar": "العربية",
    "hi": "हिन्दी",
    "th": "ไทย",
    "vi": "Tiếng Việt",
    "id": "Bahasa Indonesia",
    "ms": "Bahasa Melayu",
    "tr": "Türkçe",
    "pl": "Polski",
    "nl": "Nederlands",
    "sv": "Svenska",
    "da": "Dansk",
    "fi": "Suomi",
    "no": "Norsk",
    "cs": "Čeština",
    "hu": "Magyar",
    "el": "Ελληνικά",
    "he": "עברית",
    "uk": "Українська",
}


class LanguageDetector:
    def __init__(self):
        pass

    @staticmethod
    def get_language_name(code: str) -> str:
        code_lower = code.lower()
        for key, name in LANGUAGE_NAMES.items():
            if key.lower() == code_lower or code_lower.startswith(key.lower()):
                return name
        return code.upper()

    @staticmethod
    def get_confidence_level(probability: float) -> str:
        if probability >= 0.9:
            return "high"
        elif probability >= 0.7:
            return "medium"
        else:
            return "low"

    @staticmethod
    def is_common_multilingual(code: str) -> bool:
        common = {"zh", "en", "ja", "ko"}
        code_lower = code.lower()
        return any(code_lower.startswith(c) for c in common)
