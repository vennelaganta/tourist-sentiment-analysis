"""Print your Python / RAM / disk / GPU facts and a model recommendation.  Run: python check_system.py"""
import os
import platform
import shutil
import sys
from pathlib import Path


def main():
    print(f"Python        : {sys.version.split()[0]} ({platform.architecture()[0]})  -> need 3.10-3.12 recommended")
    print(f"OS            : {platform.platform()}")
    ram = None
    try:
        import psutil
        ram = psutil.virtual_memory().total / 2**30
        print(f"RAM           : {ram:.1f} GB total, {psutil.virtual_memory().available / 2**30:.1f} GB free now")
        print(f"CPU cores     : {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count()} logical")
    except ImportError:
        print("RAM           : install psutil (pip install -r requirements.txt) to see this")
    total, used, free = shutil.disk_usage(Path.home())
    print(f"Disk free     : {free / 2**30:.1f} GB on the drive holding {Path.home()}")
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    print(f"HF cache dir  : {cache}  (models are stored here once; set HF_HOME to move it)")
    gpu = False
    try:
        import torch
        gpu = torch.cuda.is_available()
        print(f"PyTorch       : {torch.__version__}  | CUDA GPU: {torch.cuda.get_device_name(0) if gpu else 'none (CPU mode - fine for this project)'}")
    except ImportError:
        print("PyTorch       : not installed yet")
    print("\nRecommendation (heuristic, not a benchmark):")
    if ram is not None and ram < 6:
        print("  * RAM < 6 GB  -> use the lighter model:  $env:SENTIMENT_MODEL_ID=\"clapAI/mmBERT-small-multilingual-sentiment\"")
    else:
        print("  * Default model clapAI/roberta-base-multilingual-sentiment should run on CPU (needs roughly 2-3 GB RAM).")
    if free / 2**30 < 5:
        print("  * Less than 5 GB free disk: free some space before the first model download.")
    print("  * Use --per-language 100 for the first evaluation; raise it only if speed is acceptable.")


if __name__ == "__main__":
    main()
