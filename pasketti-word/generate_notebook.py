"""Generate Kaggle training notebook for Word Track.

Reads config from environment variables (injected by GitHub Actions workflow):
  GH_PAT: GitHub PAT for artifact download
  WANDB_API_KEY: W&B API key
  RUN_MEMO: experiment description
  MODEL_NAME: base NeMo model (default: nvidia/parakeet-tdt-0.6b-v2)
  EPOCHS: training epochs (default: 20)
  BATCH_SIZE: per-device batch size (default: 16)
  GRADIENT_ACCUMULATION: gradient accumulation steps (default: 4)
  LEARNING_RATE: learning rate (default: 1e-3)
"""

import base64
import json
import os

# Config from environment
GH_PAT = os.environ.get("GH_PAT", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
RUN_MEMO = os.environ.get("RUN_MEMO", "v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v2")
EPOCHS = os.environ.get("EPOCHS", "20")
BATCH_SIZE = os.environ.get("BATCH_SIZE", "16")
GRADIENT_ACCUMULATION = os.environ.get("GRADIENT_ACCUMULATION", "4")
LEARNING_RATE = os.environ.get("LEARNING_RATE", "1e-3")
ARTIFACT_NAME = os.environ.get("ARTIFACT_NAME", "drivendata-word-data")
GH_REPO = os.environ.get("GH_REPO", "yasumorishima/drivendata-comp")
EXPORT_ONLY = os.environ.get("EXPORT_ONLY", "false").lower() == "true"

cells = []


def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [source],
    })


def add_code(source):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [source],
        "execution_count": None,
        "outputs": [],
    })


# --- Notebook cells ---

add_md("# Pasketti Word Track - Kaggle GPU Training\n\nNVIDIA Parakeet TDT 0.6B + Linear Adapter fine-tuning")

# Environment validation (MUST be first code cell — fail fast)
add_code("""import torch, os, shutil

errors = []
print("=== Environment Validation ===")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    cap_str = f"{cap[0]}.{cap[1]}"
    print(f"GPU: {gpu_name} (sm_{cap[0]}{cap[1]}, compute {cap_str})")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
    min_cap = (7, 0)
    if cap < min_cap:
        print(f"WARNING: GPU compute capability {cap_str} < {min_cap[0]}.{min_cap[1]} — may have limited PyTorch support")
    try:
        x = torch.randn(2, 2, device="cuda")
        _ = x @ x
        print("CUDA smoke test: OK")
    except Exception as e:
        errors.append(f"CUDA smoke test failed: {e}")
else:
    errors.append("No CUDA GPU detected.")

disk = shutil.disk_usage("/kaggle/working")
print(f"Disk free: {disk.free / 1024**3:.1f} GB")
print(f"PyTorch: {torch.__version__}")

if errors:
    print("\\n=== VALIDATION FAILED ===")
    for e in errors:
        print(f"  ERROR: {e}")
    raise RuntimeError(f"Environment validation failed: {'; '.join(errors)}")
print("=== Validation passed ===\\n")
""")

add_code(f"""import os
os.environ["WANDB_API_KEY"] = "{WANDB_API_KEY}"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = "drivendata-word-asr"
os.environ["GH_PAT"] = "{GH_PAT}"
print("Environment configured")
""")

add_code("""# Install dependencies
!pip install -q nemo_toolkit[asr] lightning librosa soundfile jiwer wandb requests pyyaml
import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
""")

# Embed train.py inline via base64
train_py_path = os.path.join(os.path.dirname(__file__) or ".", "train.py")
with open(train_py_path, "rb") as f:
    train_b64 = base64.b64encode(f.read()).decode()

add_code(f"""# Write train.py from embedded base64
import base64, pathlib
train_b64 = "{train_b64}"
pathlib.Path("train.py").write_bytes(base64.b64decode(train_b64))
print(f"train.py written ({{pathlib.Path('train.py').stat().st_size}} bytes)")
""")

if EXPORT_ONLY:
    # Export mode: just download pretrained model, no data needed
    add_code(f"""# Export pretrained model (no training)
import sys, runpy
sys.argv = [
    "train.py",
    "--output_dir", "/kaggle/working/model_word",
    "--model_name", "{MODEL_NAME}",
    "--export_only",
]
runpy.run_path("train.py", run_name="__main__")
""")
else:
    # Full training mode: download data, extract, train
    add_code(f"""# Download data from GitHub Artifact
import requests, io, zipfile, os

GH_PAT = os.environ["GH_PAT"]
REPO = "{GH_REPO}"
ARTIFACT_NAME = "{ARTIFACT_NAME}"

headers = {{"Authorization": f"token {{GH_PAT}}", "Accept": "application/vnd.github+json"}}

# Find latest artifact
resp = requests.get(
    f"https://api.github.com/repos/{{REPO}}/actions/artifacts",
    headers=headers, params={{"name": ARTIFACT_NAME, "per_page": 1}}
)
resp.raise_for_status()
artifacts = resp.json()["artifacts"]
assert artifacts, f"No artifact '{{ARTIFACT_NAME}}' found"
artifact = artifacts[0]
assert not artifact.get("expired"), "Artifact expired! Re-run Download Competition Data workflow."

size_gb = artifact["size_in_bytes"] / 1024**3
print(f"Downloading: {{artifact['name']}} ({{size_gb:.1f}} GB)")
print("This will take several minutes for the 17 GB word dataset...")

resp = requests.get(artifact["archive_download_url"], headers=headers, stream=True)
resp.raise_for_status()

# Stream to temp file to avoid OOM
tmp_path = "/tmp/word_data.zip"
downloaded = 0
with open(tmp_path, "wb") as f:
    for chunk in resp.iter_content(chunk_size=8192 * 1024):
        f.write(chunk)
        downloaded += len(chunk)
        if downloaded % (500 * 1024 * 1024) == 0:
            print(f"  Downloaded {{downloaded / 1024**3:.1f}} GB...")

print(f"Download complete: {{downloaded / 1024**3:.1f}} GB")

os.makedirs("data/word", exist_ok=True)
with zipfile.ZipFile(tmp_path) as zf:
    zf.extractall("data/word")
os.remove(tmp_path)

print("Extracted files:")
for f in sorted(os.listdir("data/word"))[:20]:
    size = os.path.getsize(f"data/word/{{f}}") / 1024 / 1024
    print(f"  {{f}} ({{size:.1f}} MB)")
""")

    add_code("""# Extract audio ZIP files
import zipfile
from pathlib import Path

data_dir = Path("data/word")
for zf_path in sorted(data_dir.glob("*.zip")):
    print(f"Extracting: {zf_path.name}")
    with zipfile.ZipFile(zf_path) as zf:
        zf.extractall(data_dir)
    print(f"  Done: {zf_path.name}")

# Check transcript count
import json
transcripts = [json.loads(l) for l in open(data_dir / "train_word_transcripts.jsonl")]
print(f"\\nTotal transcripts: {len(transcripts)}")
print(f"Sample: {transcripts[0]}")
""")

    add_code(f"""# Train
import sys, runpy
sys.argv = [
    "train.py",
    "--data_dir", "data/word",
    "--output_dir", "/kaggle/working/model_word",
    "--model_name", "{MODEL_NAME}",
    "--max_steps", "5000",
    "--batch_size", "{BATCH_SIZE}",
    "--lr", "{LEARNING_RATE}",
    "--eval_steps", "500",
    "--wandb_project", "drivendata-word-asr",
    "--memo", "{RUN_MEMO}",
]
runpy.run_path("train.py", run_name="__main__")
""")

add_code("""# Check output
import os

# NeMo saves as single .nemo file
model_path = "/kaggle/working/model_word/final_model.nemo"
if os.path.exists(model_path):
    size = os.path.getsize(model_path) / 1024 / 1024
    print(f"Model: {model_path} ({size:.1f} MB)")
else:
    print("ERROR: Model not found!")
    print("Contents of /kaggle/working/model_word/:")
    for f in os.listdir("/kaggle/working/model_word/"):
        fpath = os.path.join("/kaggle/working/model_word/", f)
        size = os.path.getsize(fpath) / 1024 / 1024 if os.path.isfile(fpath) else 0
        print(f"  {f} ({size:.1f} MB)")
""")

add_code("""# Package model for output
import tarfile
import shutil

model_path = "/kaggle/working/model_word/final_model.nemo"
tar_path = "/kaggle/working/model_word.tar.gz"

with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(model_path, arcname="final_model.nemo")

size_mb = os.path.getsize(tar_path) / 1024 / 1024
print(f"Model archive: {tar_path} ({size_mb:.1f} MB)")

# Copy wandb offline runs to working dir for download
import glob
wandb_dirs = glob.glob("/kaggle/working/model_word/wandb/offline-run-*")
if not wandb_dirs:
    wandb_dirs = glob.glob("/kaggle/working/wandb/offline-run-*")
for d in wandb_dirs:
    dest = f"/kaggle/working/{os.path.basename(d)}"
    shutil.copytree(d, dest, dirs_exist_ok=True)
    print(f"Copied W&B run: {dest}")

print("\\nDone! Kernel output ready for download.")
""")

# --- Build notebook ---
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = os.path.join(os.path.dirname(__file__) or ".", "train_kaggle.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Generated: {output_path} ({len(cells)} cells)")
