"""Language helpers. Detection is OPTIONAL (langdetect); the user can always choose the language."""

# The 16 language codes the chosen model was fine-tuned on (from the dataset/model cards).
SUPPORTED_LANGUAGES = {
    "ar": "Arabic", "de": "German", "en": "English", "es": "Spanish", "fr": "French",
    "hi": "Hindi", "id": "Indonesian", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "ms": "Malay", "pt": "Portuguese", "ru": "Russian", "tr": "Turkish", "vi": "Vietnamese",
    "zh": "Chinese",
}
_OTHER_NAMES = {"nl": "Dutch", "pl": "Polish", "th": "Thai", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
                "uk": "Ukrainian", "el": "Greek", "he": "Hebrew", "fa": "Persian", "sv": "Swedish",
                "da": "Danish", "no": "Norwegian", "fi": "Finnish", "cs": "Czech", "ro": "Romanian",
                "hu": "Hungarian", "bg": "Bulgarian", "ur": "Urdu", "mr": "Marathi", "gu": "Gujarati",
                "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "ne": "Nepali", "sw": "Swahili",
                "tl": "Tagalog", "ca": "Catalan", "hr": "Croatian", "sk": "Slovak", "sl": "Slovenian"}
_CODE_FIX = {"zh-cn": "zh", "zh-tw": "zh"}


def language_name(code):
    if not code:
        return "Unknown"
    return SUPPORTED_LANGUAGES.get(code) or _OTHER_NAMES.get(code) or code.upper()


def detection_available() -> bool:
    try:
        import langdetect  # noqa: F401
        return True
    except ImportError:
        return False


def detect_language(text):
    """Return (iso_code, probability) or (None, None) when it cannot be determined."""
    try:
        from langdetect import DetectorFactory, detect_langs
        from langdetect.lang_detect_exception import LangDetectException
    except ImportError:
        return None, None
    DetectorFactory.seed = 0  # deterministic results
    if sum(ch.isalpha() for ch in text) < 3:
        return None, None
    try:
        best = detect_langs(text)[0]
    except LangDetectException:
        return None, None
    code = _CODE_FIX.get(best.lang.lower(), best.lang.lower())
    return code, float(best.prob)
