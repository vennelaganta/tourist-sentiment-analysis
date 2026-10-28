"""Multilingual sentiment engine: a pretrained transformer used for INFERENCE ONLY.

Default model: clapAI/roberta-base-multilingual-sentiment (XLM-RoBERTa-base fine-tuned by clapAI on
clapAI/MultiLingualSentiment). Nothing is trained when the website starts; the model is downloaded once
from the Hugging Face Hub and then served from the local cache. No token/credential is required.
"""
import os
import threading

from preprocessing import LABELS, normalize_label

DEFAULT_MODEL_ID = "clapAI/roberta-base-multilingual-sentiment"
LIGHTER_MODEL_ID = "clapAI/mmBERT-small-multilingual-sentiment"  # optional, 140M params, needs transformers>=4.55


def labels_from_config(id2label):
    """Map the model's id2label to Positive/Negative/Neutral, in index order. Fails loudly if it cannot."""
    mapped = []
    for idx in sorted(int(k) for k in id2label):
        raw = id2label[idx] if idx in id2label else id2label[str(idx)]
        label = normalize_label(raw) if not str(raw).isdigit() else None
        if label not in LABELS:
            raise ValueError(f"Model label '{raw}' is not Positive/Negative/Neutral; "
                             f"full id2label = {dict(id2label)}. Refusing to guess a mapping.")
        mapped.append(label)
    if len(set(mapped)) != len(mapped):
        raise ValueError(f"Duplicate labels after mapping: {mapped}")
    return mapped


class TransformersBackend:
    """Loads a Hugging Face sequence-classification model and predicts in batches."""

    def __init__(self, model_id=None, max_tokens=256, batch_size=16, local_only=False):
        self.model_id = model_id or os.environ.get("SENTIMENT_MODEL_ID", DEFAULT_MODEL_ID)
        self.max_tokens, self.batch_size, self.local_only = max_tokens, batch_size, local_only
        self.device = "cpu"
        self.labels = []
        self._model = self._tok = self._torch = None
        self.token_limit = max_tokens

    def load(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        kw = {"local_files_only": self.local_only}
        self._tok = AutoTokenizer.from_pretrained(self.model_id, **kw)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id, **kw)
        self.labels = labels_from_config(model.config.id2label)
        if self.device == "cuda":
            model = model.half()  # fp16 only on GPU; CPU stays fp32 for speed/accuracy
        self._model = model.to(self.device).eval()
        model_max = getattr(self._tok, "model_max_length", 512)
        self.token_limit = min(self.max_tokens, model_max if model_max < 100000 else 512)

    def predict(self, texts):
        torch, out = self._torch, []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            lengths = [len(ids) for ids in self._tok(chunk, truncation=False)["input_ids"]]
            enc = self._tok(chunk, padding=True, truncation=True, max_length=self.token_limit,
                            return_tensors="pt").to(self.device)
            with torch.inference_mode():
                probs = torch.softmax(self._model(**enc).logits.float(), dim=-1).cpu().tolist()
            for p, n in zip(probs, lengths):
                scores = {lab: float(v) for lab, v in zip(self.labels, p)}
                scores = {l: scores[l] for l in LABELS if l in scores}
                best = max(scores, key=scores.get)
                out.append({"label": best, "probabilities": scores, "confidence": scores[best],
                            "n_tokens": n, "truncated": n > self.token_limit})
        return out

    def info(self):
        return {"model_id": self.model_id, "device": self.device, "labels": self.labels,
                "max_tokens": self.token_limit}


def describe_load_error(exc):
    msg = str(exc)[:300]
    if isinstance(exc, ImportError):
        return f"A required library is missing ({msg}). Run: pip install -r requirements.txt"
    low = msg.lower()
    if any(k in low for k in ("connect", "offline", "resolve", "max retries", "timed out", "couldn't find", "not a local folder")):
        return ("The model could not be downloaded (check your internet connection, then retry). "
                f"After one successful download it works offline. Details: {msg}")
    return f"Model failed to load: {msg}"


class ModelManager:
    """Thread-safe lazy loader around a backend; status: not_loaded | loading | ready | error."""

    def __init__(self, backend):
        self.backend, self.status, self.error = backend, "not_loaded", None
        self._lock = threading.Lock()

    def _claim(self):
        with self._lock:
            if self.status in ("loading", "ready"):
                return False
            self.status, self.error = "loading", None
            return True

    def _do_load(self):
        try:
            self.backend.load()
            self.status = "ready"
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, never swallowed silently
            self.error, self.status = describe_load_error(exc), "error"

    def load_sync(self):
        if self._claim():
            self._do_load()

    def load_background(self):
        if self._claim():
            threading.Thread(target=self._do_load, daemon=True).start()

    def predict(self, texts):
        return self.backend.predict(texts)
