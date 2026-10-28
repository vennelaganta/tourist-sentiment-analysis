# Tourist Sentiment Analysis — Multilingual Edition

Tourist Place Reviews Sentiment Classification Using Machine Learning.
Type a review in **any of 16 supported languages** → the website returns **Positive / Negative / Neutral**, a **confidence score**, and the **language**.

* Dataset: [clapAI/MultiLingualSentiment](https://huggingface.co/datasets/clapAI/MultiLingualSentiment) (streamed with Hugging Face `datasets`, never fully downloaded)
* Model: [clapAI/roberta-base-multilingual-sentiment](https://huggingface.co/clapAI/roberta-base-multilingual-sentiment) — XLM-RoBERTa-base (278M parameters) already fine-tuned by clapAI on that dataset. Used for **inference only**; nothing is trained on your laptop.
* Backend: Flask + PyTorch (CPU is fine) · Frontend: your existing pages (Home, Analyze, Dashboard, Dataset, About) with the same design.

---
## 0. What is verified and what is not
| Item | Status |
|---|---|
| Flask API, all 5 pages, validation, history, dashboard counts, error handling (tested with a clearly labelled **test double** instead of the real model) | ✅ tested (41 tests passed, 1 optional test skipped) |
| Sampling, de-duplication, metric formulas (hand-checked example), label mapping, language tables | ✅ tested |
| `evaluate_model.py` output file matches what the Dashboard reads (checked with stub data) | ✅ checked |
| `pip install` of `requirements.txt`, Hugging Face downloads, dataset streaming, **real model predictions, speed, and all accuracy/F1 numbers** | ⚠️ **NOT run by me** (the authoring sandbox had no internet and no PyTorch). You run them in sections 5–8. |

**No evaluation results are shipped.** The Dashboard shows empty states until *you* run `evaluate_model.py`. The F1 scores printed on the Hugging Face model cards (e.g. 81.8 for this model) are the authors' numbers, not ours, and are not used anywhere in the website.

---
## 1. Extract and open in VS Code
```powershell
cd $HOME\Downloads
Expand-Archive .\tourist-sentiment-analysis.zip -DestinationPath . -Force
cd .\tourist-sentiment-analysis
code .
```
In VS Code: **Terminal → New Terminal** (PowerShell). Run all commands below from this folder.

## 2. Check your computer
```powershell
python --version                      # need 3.10, 3.11 or 3.12 (64-bit)
Get-CimInstance Win32_ComputerSystem | Select-Object @{n='RAM_GB';e={[math]::Round($_.TotalPhysicalMemory/1GB,1)}}
Get-PSDrive C | Select-Object @{n='Free_GB';e={[math]::Round($_.Free/1GB,1)}}
nvidia-smi                            # optional: only works if you have an NVIDIA GPU
```
**Rough requirements (estimates, not measurements):**

| Resource | Needed | Notes |
|---|---|---|
| Python | 3.10–3.12, 64-bit | Python 3.13 may lack wheels for some libraries |
| RAM | ≥ 8 GB comfortable; ~4 GB minimum | Model ≈ 1.1 GB in memory (fp32 on CPU) + Python/PyTorch overhead |
| Disk | ~5 GB free | Libraries ≈ 1.5–2.5 GB, model download ≈ 0.6 GB (278M params × 2 bytes) |
| Internet | ≈ 0.5–1 GB for libraries + ≈ 0.6 GB model, **once** | Model is cached afterwards; works offline |
| Dataset | **streamed**, nothing stored | Whole dataset is 988 MB per its card; a test-split pass reads only ~100 MB at most |
| GPU | **not required** | A CUDA GPU is used automatically if PyTorch can see one |

Low on RAM (< 6 GB)? Use the lighter model later: `$env:SENTIMENT_MODEL_ID="clapAI/mmBERT-small-multilingual-sentiment"` (140M parameters, needs `transformers>=4.55`, already in requirements).

## 3. Virtual environment and dependencies
```powershell
python -m venv venv
venv\Scripts\Activate.ps1            # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
python check_system.py                # prints your RAM / disk / GPU and a model recommendation
```
`requirements.txt` uses version **ranges** (I could not run pip to lock exact versions). After it works, save your exact set: `pip freeze > requirements-lock.txt`.

## 4. Load and inspect the dataset (reads the real schema)
```powershell
python inspect_dataset.py --quick     # 2,000-row streamed sample, ~1 minute
python inspect_dataset.py             # optional: 20,000-row sample
```
This prints the real columns (the dataset card lists `text, label, source, domain, language`), label values, languages, and domains, and writes `data\dataset_profile.json`, which the **Dataset** page then displays. Counts come from the sample and are labelled as such.

## 5. Small first test — does the real model work?
```powershell
$env:RUN_MODEL_TESTS="1"
python -m pytest tests\test_model_integration.py -q
Remove-Item Env:RUN_MODEL_TESTS
```
First run downloads the model (~0.6 GB). The test loads it, checks the label mapping, and classifies six example sentences from the model card in English, German, Spanish, Japanese and Arabic. **If a test fails, send me the output** — that is exactly what I could not check.

Always-available tests (no model needed): `python -m pytest -q`

## 6. Training / fine-tuning
**Not needed for this version.** The chosen model was fine-tuned by its authors on this dataset's training split on multi-GPU hardware. Doing that on a laptop is impractical (3.1M training reviews).
*Optional upgrade paths:* (a) `clapAI/roberta-large-multilingual-sentiment` (560M) on a GPU; (b) fine-tune on only the tourism-like domains using Hugging Face `Trainer` on a free Colab/Kaggle GPU, then point `SENTIMENT_MODEL_ID` at the saved folder; (c) INT8 quantisation/ONNX for faster CPU inference.

## 7. Evaluate the model (measured results for the Dashboard)
Method: results are computed on a **language-stratified streamed sample**, not the full 393k-review split. Validation is for choosing between models; **test is used for the final report only** (don't tune against it).
```powershell
# 7a. compare candidates on VALIDATION (≈1,600 reviews each; the script prints reviews/second so you can plan bigger runs)
python evaluate_model.py --split validation --per-language 100
python evaluate_model.py --split validation --per-language 100 --model-id clapAI/mmBERT-small-multilingual-sentiment

# 7b. final report on TEST with your chosen model (run once)
python evaluate_model.py --split test --per-language 100

# 7c. optional extras
python evaluate_model.py --split test --per-language 100 --domains "hotel,restaurant,places"   # tourist-like domains only
python evaluate_model.py --split test --per-language 100 --balance-classes                    # equal class counts
python evaluate_model.py --split test --per-language 100 --overlap-scan 200000                # count eval texts also found in 200k train rows
```
A quick smoke run: `python evaluate_model.py --split validation --per-language 20`.
Check domain names first in `data\dataset_profile.json` and adjust `--domains` if they differ.
Reported: accuracy, **macro F1**, weighted F1, balanced accuracy, per-class precision/recall/F1, confusion matrix, per-language and per-domain scores, sample size and language coverage. Languages with < 100 samples are flagged *low n*. Rare languages (Italian, Malay, Hindi) can take a full pass over the split to fill their quota — that is normal; the sample counts are shown. Files are saved as `models\evaluation_<split>_<model>*.json`.

**Class imbalance:** the card reports Negative 1.44M / Neutral 1.04M / Positive 1.46M, so Neutral is the smallest class. We rely on macro F1, balanced accuracy and per-class recall rather than accuracy alone, and offer `--balance-classes`.

## 8. Start the website
```powershell
python app.py
```
Open **http://127.0.0.1:5000** in Chrome/Edge. The model loads in the background (footer pill: *Loading multilingual model…* → *Ready*). Analyze requests made while it loads get a friendly "still loading" message. Stop with `Ctrl+C`.

Useful environment variables (set before `python app.py`):
```powershell
$env:SENTIMENT_MODEL_ID="clapAI/mmBERT-small-multilingual-sentiment"   # use another model
$env:SENTIMENT_MAX_TOKENS="256"       # truncate longer reviews (max 512 for the default model)
$env:SENTIMENT_LOCAL_ONLY="1"         # offline: use only the cached model
$env:HF_HOME="D:\hf_cache"            # move the model cache to another drive
```
No Hugging Face token is needed; never paste a token into the code.

## 9. Using the site
* **Analyze** – choose a language or leave *Auto-detect*; try the multilingual examples. You see the original review, label, confidence, probabilities, detected/selected language, and warnings (unsupported language, truncation, uncertain detection).
* **Dashboard** – counts, language chart, history (search/filters) from your real predictions in `data\predictions.csv`, then the evaluation runs you generated.
* **Dataset** – real Hugging Face dataset profile, plus a CSV inspector.
* Languages: Arabic, German, English, Spanish, French, Hindi, Indonesian, Italian, Japanese, Korean, Malay, Portuguese, Russian, Turkish, Vietnamese, Chinese. Other languages are **not translated**; they are still scored but flagged as less reliable.

## 10. Design decisions (for your viva)
* **Why a transformer?** Classical TF-IDF models are tied to one language/script; a multilingual transformer shares one vocabulary and representation across languages. Your NB/SVM/RF code is kept in `classical_baseline\` as optional background and is not used by the website.
* **Why this model?** It was fine-tuned on exactly this dataset (no domain mismatch with the data you evaluate on), is XLM-R-based (mature, widely supported), and fits CPU laptops. `mmBERT-small` (140M, authors report slightly higher F1) is a lighter alternative but needs a newer `transformers`; both are one setting apart. Reported author F1: roberta-base 81.8, mmBERT-small 82.2, roberta-large 82.6 (their numbers, averaging not stated).
* **No translation** of reviews: it adds latency, cost and errors; the original text is preserved and shown.
* **Consistent preprocessing:** `preprocessing.prepare_text` (whitespace/control-char cleanup only) is used for both evaluation and live predictions.
* **Honest limits:** dataset is not tourism-only (products, movies, tweets…), has label noise, and low-resource languages (Hindi ~9k, Malay ~7k, Italian ~3k records vs English ~1.5M per the card) are likely weaker. LSTM/BiLSTM are discussed in the About page but **not implemented**.

## 11. Troubleshooting
| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `pip install` fails building a package | Use Python 3.11/3.12 64-bit; `python -m pip install --upgrade pip` |
| `ModuleNotFoundError` | venv not active (prompt should start with `(venv)`) or run `pip install -r requirements.txt` |
| Footer says **Model error** / Analyze says *could not be downloaded* | Check internet/proxy, then click Analyze again after `Invoke-RestMethod -Method Post http://127.0.0.1:5000/api/model/reload` or restart the app |
| Download interrupted / corrupted cache | `Remove-Item -Recurse $HOME\.cache\huggingface\hub\models--clapAI--roberta-base-multilingual-sentiment` then retry |
| `Could not stream the dataset` | Network issue; retry. For slow links: `$env:HF_HUB_DOWNLOAD_TIMEOUT="60"` |
| Hugging Face "symlinks" warning on Windows | Harmless; silence with `$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"` |
| Error mentioning `sentencepiece` | `pip install sentencepiece` |
| Error about unknown model architecture (mmBERT) | `pip install -U "transformers>=4.55,<5"` |
| `Refusing to guess a mapping` (labels) | The model's `id2label` isn't positive/negative/neutral; use a different model |
| Out of memory / very slow | Close other apps; use the mmBERT-small model; `$env:SENTIMENT_MAX_TOKENS="128"` |
| Port 5000 busy | Edit the last line of `app.py` (`port=5001`) |
| Language shown as Unknown / wrong | Reviews < ~3 letters can't be detected; pick the language manually (detection uses `langdetect`) |
| Dashboard: "No evaluation results" | Run section 7 |

## 12. Project structure
```
app.py                      Flask app + JSON API (/api/health, predict, dashboard, history, metrics, dataset-profile, upload-dataset, dataset-summary, model/reload)
sentiment_engine.py         transformer backend, label mapping, thread-safe lazy loader
language_utils.py           supported languages + optional detection
preprocessing.py            prepare_text (live + evaluation) and label helpers
evaluation_utils.py         sampling, de-duplication, metrics (unit-tested)
evaluate_model.py           streamed, language-stratified evaluation -> models\evaluation_*.json
inspect_dataset.py          dataset schema/distribution profile -> data\dataset_profile.json
check_system.py             Python/RAM/disk/GPU report
classical_baseline\         optional old TF-IDF + NB/SVM/RF CSV trainer (not used by the website)
templates\, static\         your existing pages, CSS and JS (updated, same design)
tests\                      test_app, test_core, test_classical_baseline, test_model_integration (opt-in)
```
