"""Text preprocessing + label helpers shared by train_model.py and app.py.

The SAME clean_text function is used for training and for predicting new reviews,
so there is no train/serve mismatch. Stop-words are NOT removed on purpose:
words like "not" and "no" carry sentiment ("not good" != "good").
"""
import re

import pandas as pd

LABELS = ["Positive", "Negative", "Neutral"]
_URL = re.compile(r"https?://\S+|www\.\S+")
_NON_ALPHA = re.compile(r"[^a-z\s']")


def clean_text(text) -> str:
    """Lowercase, strip URLs/digits/punctuation, expand n't, squeeze spaces."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    t = str(text).lower()
    t = _URL.sub(" ", t)
    t = t.replace("n't", " not")
    t = _NON_ALPHA.sub(" ", t)
    t = t.replace("'", " ")
    return re.sub(r"\s+", " ", t).strip()


_STRING_LABELS = {
    "positive": "Positive", "pos": "Positive", "good": "Positive",
    "negative": "Negative", "neg": "Negative", "bad": "Negative",
    "neutral": "Neutral", "neu": "Neutral", "mixed": "Neutral",
}
_RATING_LABELS = {1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"}


def normalize_label(value, custom_map=None):
    """Map a raw label (string or 1-5 rating) to Positive/Negative/Neutral, else None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    key = str(value).strip().lower()
    if custom_map and key in custom_map:
        return custom_map[key]
    if key in _STRING_LABELS:
        return _STRING_LABELS[key]
    try:
        return _RATING_LABELS.get(int(float(key)))
    except ValueError:
        return None


def parse_label_map(text):
    """'0=Negative,1=Neutral,2=Positive' -> {'0':'Negative', ...}"""
    mapping = {}
    for pair in (text or "").split(","):
        if "=" in pair:
            raw, target = pair.split("=", 1)
            target = target.strip().title()
            if target not in LABELS:
                raise ValueError(f"Label map target must be one of {LABELS}, got '{target}'")
            mapping[raw.strip().lower()] = target
    return mapping


def guess_columns(columns):
    """Guess (text_column, label_column) from column names."""
    def find(words):
        for w in words:
            for c in columns:
                if w in str(c).lower():
                    return c
        return None
    return (find(["review", "text", "comment", "feedback", "content"]),
            find(["sentiment", "label", "class", "polarity", "rating"]))


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def prepare_text(text) -> str:
    """Minimal normalisation for the TRANSFORMER path (live predictions AND evaluation).

    No lower-casing / punctuation stripping / stop-word removal: the multilingual tokenizer
    handles case, scripts and emoji itself, and aggressive cleaning would damage non-Latin text.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return re.sub(r"\s+", " ", _CONTROL.sub(" ", str(text))).strip()
