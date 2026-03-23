"""Generate Kaggle training notebook for Phonetic Track.

Reads config from environment variables (injected by GitHub Actions workflow):
  GH_PAT: GitHub PAT for artifact download
  WANDB_API_KEY: W&B API key
  RUN_MEMO: experiment description
  MODEL_NAME: base model (default: facebook/wav2vec2-base)
  EPOCHS: training epochs (default: 20)
  BATCH_SIZE: per-device batch size (default: 16)
  GRADIENT_ACCUMULATION: gradient accumulation steps (default: 4)
  LEARNING_RATE: learning rate (default: 5e-5)
"""

import json
import os

# Config from environment
GH_PAT = os.environ.get("GH_PAT", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
RUN_MEMO = os.environ.get("RUN_MEMO", "v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "facebook/wav2vec2-base")
EPOCHS = os.environ.get("EPOCHS", "20")
BATCH_SIZE = os.environ.get("BATCH_SIZE", "8")
GRADIENT_ACCUMULATION = os.environ.get("GRADIENT_ACCUMULATION", "8")
LEARNING_RATE = os.environ.get("LEARNING_RATE", "5e-5")
ARTIFACT_NAME = os.environ.get("ARTIFACT_NAME", "drivendata-phonetic-data")
GH_REPO = os.environ.get("GH_REPO", "yasumorishima/drivendata-comp")

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

add_md("# Pasketti Phonetic Track - Kaggle GPU Training")

add_code(f"""import os
os.environ["WANDB_API_KEY"] = "{WANDB_API_KEY}"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = "drivendata-phonetic-asr"
os.environ["GH_PAT"] = "{GH_PAT}"
print("Environment configured")
""")

add_code("""# Install dependencies
!pip install -q transformers[torch] datasets librosa soundfile jiwer wandb requests
import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
""")

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

print(f"Downloading: {{artifact['name']}} ({{artifact['size_in_bytes'] / 1024 / 1024:.1f}} MB)")
resp = requests.get(artifact["archive_download_url"], headers=headers)
resp.raise_for_status()

os.makedirs("/tmp/data/phonetic", exist_ok=True)
with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
    zf.extractall("/tmp/data/phonetic")

print("Downloaded files:")
for f in sorted(os.listdir("/tmp/data/phonetic"))[:20]:
    size = os.path.getsize(f"/tmp/data/phonetic/{{f}}") / 1024 / 1024
    print(f"  {{f}} ({{size:.1f}} MB)")
""")

add_code("""# Extract audio ZIP files
import zipfile
from pathlib import Path

data_dir = Path("/tmp/data/phonetic")
for zf_path in sorted(data_dir.glob("*.zip")):
    print(f"Extracting: {zf_path.name}")
    with zipfile.ZipFile(zf_path) as zf:
        zf.extractall(data_dir)
    print(f"  Done: {zf_path.name}")

# Check transcript count
import json
transcripts = [json.loads(l) for l in open(data_dir / "train_phon_transcripts.jsonl")]
print(f"\\nTotal transcripts: {len(transcripts)}")
print(f"Sample: {transcripts[0]}")
""")

# Embed train.py inline via base64 (no git clone dependency)
import base64
train_py_path = os.path.join(os.path.dirname(__file__) or ".", "train.py")
with open(train_py_path, "rb") as f:
    train_b64 = base64.b64encode(f.read()).decode()

add_code(f"""# Write train.py from embedded base64
import base64, pathlib
train_b64 = "{train_b64}"
pathlib.Path("train.py").write_bytes(base64.b64decode(train_b64))
print(f"train.py written ({{pathlib.Path('train.py').stat().st_size}} bytes)")
""")

add_code(f"""# Train (runpy — same process, isolated namespace, shares TPU device)
import sys, runpy
sys.argv = [
    "train.py",
    "--data_dir", "/tmp/data/phonetic",
    "--output_dir", "/kaggle/working/model_phonetic",
    "--model_name", "{MODEL_NAME}",
    "--epochs", "{EPOCHS}",
    "--batch_size", "{BATCH_SIZE}",
    "--gradient_accumulation", "{GRADIENT_ACCUMULATION}",
    "--lr", "{LEARNING_RATE}",
    "--wandb_project", "drivendata-phonetic-asr",
    "--memo", "{RUN_MEMO}",
]
runpy.run_path("train.py", run_name="__main__")
""")

add_code("""# Check output
import os
model_dir = "/kaggle/working/model_phonetic/final_model"
if os.path.exists(model_dir):
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, dn, fn in os.walk(model_dir) for f in fn)
    print(f"Model directory: {model_dir}")
    print(f"Total size: {total / 1024 / 1024:.1f} MB")
    for f in sorted(os.listdir(model_dir)):
        size = os.path.getsize(os.path.join(model_dir, f)) / 1024 / 1024
        print(f"  {f} ({size:.1f} MB)")
else:
    print("ERROR: Model directory not found!")
    print("Contents of /kaggle/working/:")
    for f in os.listdir("/kaggle/working/"):
        print(f"  {f}")
""")

add_code("""# Package model for output + clean up /kaggle/working/
import tarfile
import shutil
import glob

model_dir = "/kaggle/working/model_phonetic/final_model"
tar_path = "/kaggle/working/model_phonetic.tar.gz"

with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(model_dir, arcname=".")

size_mb = os.path.getsize(tar_path) / 1024 / 1024
print(f"Model archive: {tar_path} ({size_mb:.1f} MB)")

# Copy wandb offline runs to working dir for download
wandb_dirs = glob.glob("/kaggle/working/model_phonetic/checkpoints/wandb/offline-run-*")
if not wandb_dirs:
    wandb_dirs = glob.glob("/kaggle/working/wandb/offline-run-*")
for d in wandb_dirs:
    dest = f"/kaggle/working/{os.path.basename(d)}"
    shutil.copytree(d, dest, dirs_exist_ok=True)
    print(f"Copied W&B run: {dest}")

# Clean up: remove everything except tar.gz, wandb runs, and log from /kaggle/working/
# This prevents input data or checkpoints from polluting kernel output
keep_prefixes = ("model_phonetic.tar.gz", "offline-run-")
for item in os.listdir("/kaggle/working/"):
    path = os.path.join("/kaggle/working/", item)
    if any(item.startswith(p) for p in keep_prefixes):
        continue
    if item == "model_phonetic":
        shutil.rmtree(path)
        print(f"Cleaned up: {item}/")
    elif os.path.isdir(path) and item not in ("__notebook_source__",):
        shutil.rmtree(path)
        print(f"Cleaned up: {item}/")

print("\\nFinal /kaggle/working/ contents:")
for item in sorted(os.listdir("/kaggle/working/")):
    path = os.path.join("/kaggle/working/", item)
    if os.path.isfile(path):
        print(f"  {item} ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")
    else:
        print(f"  {item}/")

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
