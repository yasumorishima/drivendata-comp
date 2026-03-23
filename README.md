# drivendata-comp

[DrivenData](https://www.drivendata.org/) competition pipeline — automated training, packaging, and submission.

---

## Current Competition: On Top of Pasketti

**Children's Speech Recognition Challenge** — $120,000 prize pool, deadline 2026-04-06

Children's speech differs significantly from adult speech (pronunciation errors, disfluencies, unique vocabulary). This competition aims to build robust ASR models that work well on children's audio.

| | Phonetic Track ($50K) | Word Track ($70K) |
|---|---|---|
| Task | IPA phonetic transcription | Word-level transcription |
| Metric | CER (Character Error Rate) | WER (Word Error Rate) |
| Benchmark | Wav2Vec2 (HuggingFace) | Parakeet TDT 0.6B (NeMo) |
| Benchmark Score | CER 0.33 | WER 0.164 |
| Data | 1.4 GB audio | 17.3 GB audio + noise |
| Submission | Code (ZIP, Docker, GPU) | Code (ZIP, Docker, GPU) |

### Approach

**Phonetic Track (Kaggle GPU / Colab T4 training)**
- Base model: `facebook/wav2vec2-base` with CTC head
- IPA vocabulary built from training transcripts (52 IPA characters)
- Training designed for **30-minute completion** on both Colab T4 and Kaggle P100
- Raw PyTorch training loop with crash-proof measures:
  - Gradient checkpointing (GPU ~40% VRAM savings)
  - `get_safe_batch_size` auto-selects batch size by VRAM (base models: batch=16 on 12GB+)
  - `mask_time_prob=0.0` to prevent crash on short audio
  - Periodic GC + `torch.cuda.empty_cache()` every 100 steps
  - GPU memory logging at checkpoints
- Checkpoint resume & crash recovery:
  - `--resume_from`: resume training from a saved checkpoint
  - `--drive_checkpoint_dir`: backup checkpoints to Google Drive
  - `--save_every_steps`: periodic checkpoint save (default: 500 steps)
- W&B experiment tracking is optional (runs without `WANDB_API_KEY`)
- **Colab workflow** (`train_colab.ipynb`): single-session completion (~30-60min on T4)
  - 5 epochs, batch=16, grad_accum=4
  - Drive data cache + auto-resume + final model save to Drive
- Previous: Colab T4 reached CER 0.535 (child-exp000, 20 epochs) but model lost to Drive sync failure
- TPU approach abandoned (v5-v27: OOM/idle timeout issues)

**Word Track (not yet started)**
- Base model: NVIDIA Parakeet TDT 0.6B (NeMo)
- Baseline: pretrained model export without fine-tuning (WER ~0.164)
- Noise augmentation using provided classroom noise samples (planned)

---

## Pipeline Overview

### Training & Submission (GitHub Actions + Kaggle GPU/TPU)

```
Download Data (Playwright) → Kaggle GPU/TPU Train → GitHub Release → Package ZIP → Submit
```

- GPU→CPU auto-fallback on quota/OOM/CUDA errors
- `COMPLETE_EMPTY` (no output) treated as failure
- Export-only mode: download pretrained model without training for baseline submission

## Workflows

| Workflow | Purpose | Trigger |
|---|---|---|
| **Check Competitions** | List active DrivenData competitions | `workflow_dispatch` |
| **Download Competition Data** | Playwright auto-download → Artifact | `workflow_dispatch` |
| **DrivenData Train & Validate** | CSV submission: train → validate → submission.csv | `workflow_dispatch` |
| **DrivenData GPU Train (Kaggle)** | Code submission: Kaggle P100 train → Release (`contents: write`) | `workflow_dispatch` |
| **DrivenData TPU Train (Kaggle)** | Code submission: Kaggle TPU v3-8 train → Release | `workflow_dispatch` |
| **Package DrivenData Submission** | Code submission: Release model + main.py → ZIP | `workflow_dispatch` |

## Quick Start

```bash
# 1. Download data
gh workflow run "Download Competition Data" \
  -f memo="initial download"

# 2a. Train on Kaggle GPU
gh workflow run "DrivenData GPU Train (Kaggle)" \
  -f competition_dir=pasketti-phonetic \
  -f model_release_tag=phonetic-model-v3 \
  -f memo="wav2vec2-base CTC baseline"

# 2b. Or train on Kaggle TPU (separate quota from GPU)
gh workflow run "DrivenData TPU Train (Kaggle)" \
  -f competition_dir=pasketti-phonetic \
  -f model_release_tag=phonetic-tpu-v1 \
  -f memo="wav2vec2-base CTC on TPU v3-8"

# 2c. Export pretrained model without training (baseline)
gh workflow run "DrivenData GPU Train (Kaggle)" \
  -f competition_dir=pasketti-word \
  -f model_release_tag=word-baseline-v1 \
  -f memo="Parakeet TDT pretrained baseline" \
  -f export_only=true

# 3. Package for submission
gh workflow run "Package DrivenData Submission" \
  -f competition_dir=pasketti-phonetic \
  -f model_release_tag=phonetic-model-v3 \
  -f memo="baseline submission"
```

## Training Flows

GPU/TPU枠は別カウント。一方が切れてももう一方で学習可能。

| Flow | Accelerator | Use Case | Quota |
|---|---|---|---|
| **Kaggle GPU** | P100 | 本番学習 / モデルエクスポート | 週30h（土曜リセット） |
| **Kaggle TPU** | TPU v3-8 (128GB HBM, bf16) | GPU枠切れ時の代替 | 週30h（GPU枠とは別） |
| **Colab GPU** | T4/A100 via [colab-mcp](https://github.com/googlecolab/colab-mcp) | 単セッション完結学習（~30-60min on T4） | 1日数時間 |

**Training strategy**: 30分以内で完走する設計（Colab T4 / Kaggle P100 両対応）。`train_colab.ipynb` はDriveデータキャッシュ + 自動リジューム + 最終モデルDrive保存で、セッション切断に強い。

### TPU Training Notes

- Kaggle TPU環境のプリインストール済み `torch_xla` をそのまま使用（再インストールするとtorchバージョン競合を起こす）
- `XLA_USE_BF16=1` + `PJRT_DEVICE=TPU` で bf16 学習を有効化
- pip installで `transformers[torch]` ではなく `transformers` を使用（`[torch]` extraがtorchを上書きしてPyTreeSpec互換性を壊す）
- HuggingFace TrainerはTPUを自動検出

### Pre-push Smoke Test (dry-run)

GPU/TPUワークフローはKaggle pushの前にCPU上でdry-runを実行する。ダミーデータで1回forward+backwardし、import・dtype・shape エラーを30秒で検出する。

```bash
# ローカルでも実行可能
python train.py --dry_run

# チェックポイントから再開（Drive保存併用）
python train.py --resume_from /path/to/checkpoint \
  --drive_checkpoint_dir /content/drive/MyDrive/pasketti/checkpoints \
  --save_every_steps 200
```

- `train.py` に `dry_run()` 関数を実装（ダミー vocab + 1秒音声 × 2サンプル）
- ワークフローの `Dry-run smoke test (CPU)` ステップで自動実行
- **新しい train.py を作る際は必ず `--dry_run` 対応を入れる**

## Tech Stack

| Component | Technology |
|---|---|
| Training | PyTorch + HuggingFace Transformers / NeMo |
| ASR Model | Wav2Vec2 (CTC) / Parakeet TDT 0.6B (NeMo) |
| Accelerator | Kaggle P100 (GPU) / Kaggle TPU v3-8 / Colab GPU ([colab-mcp](https://github.com/googlecolab/colab-mcp)) |
| CI/CD | GitHub Actions |
| Experiment Tracking | W&B (offline sync, optional) |
| Notifications | Discord Webhook |
| Data Download | Playwright (headless browser) |

## Project Structure

```
drivendata-comp/
├── .github/workflows/        # GitHub Actions pipelines
├── template/                 # CSV submission template
├── gpu-template/             # Code submission template (GPU)
├── pasketti-phonetic/        # Phonetic Track: Wav2Vec2 CTC
│   ├── train.py              # Training script (Kaggle GPU / Colab T4)
│   ├── train_colab.ipynb     # Colab notebook (single-session ~30-60min)
│   ├── main.py               # Inference script (DrivenData runtime)
│   ├── generate_notebook.py  # Embeds train.py into Kaggle notebook
│   └── kernel-metadata.json
├── pasketti-word/            # Word Track: Parakeet TDT 0.6B + Adapter
│   ├── train.py              # Training script (--export_only for baseline)
│   ├── main.py               # Inference script (DrivenData runtime)
│   ├── generate_notebook.py  # Embeds train.py into Kaggle notebook
│   └── kernel-metadata.json
├── scripts/                  # Utility scripts
└── DRIVENDATA_MEMO.md        # Internal operation notes
```

## Roadmap

- [x] Pipeline: Download → Kaggle GPU Train → Release (GPU→CPU fallback)
- [x] EXP + child-exp experiment iteration via Colab + Google Drive
- [x] Runner v4.1: real-time logging, GPU enforcement, GH_TOKEN Drive fallback, heartbeat
- [x] CDP keepalive: Chrome DevTools Protocol via systemd (replaces xdotool)
- [x] Kaggle TPU v3-8 training workflow (GPU枠とは別枠で学習可能)
- [x] Pre-push dry-run smoke test (CPU上でdtype/shape/importエラーを事前検出)
- [x] Word Track pipeline: kernel-metadata + generate_notebook + export_only mode
- [x] colab-mcp integration: Claude Code → Colab GPU direct execution for experiment iteration
- [x] Crash-proof training: gradient checkpointing, auto batch size, mask_time_prob, GC, OOM handler
- [x] Fix: data to /tmp to prevent Kaggle output pollution (12K+ files in /kaggle/working/)
- [x] Checkpoint resume & Drive backup (`--resume_from`, `--drive_checkpoint_dir`, `--save_every_steps`)
- [x] W&B made optional (runs without WANDB_API_KEY)
- [x] Colab single-session workflow (`train_colab.ipynb`): 5 epochs, batch=16, ~30-60min on T4
- [x] kaggle-train.yml `permissions: contents: write` (fixes 403 on release creation)
- [ ] Phonetic wav2vec2-base CTC — Kaggle GPU学習中 (v4)
- [ ] Word Track baseline — pretrained Parakeet TDT export
- [ ] Package Submission → first DrivenData submission (both tracks)
- [ ] Phonetic improvements: data augmentation, pyctcdecode LM
- [ ] Word Track: adapter fine-tuning with noise augmentation

## Profile

- DrivenData: [Ymori](https://www.drivendata.org/users/Ymori/)
- Kaggle: [yasunorim](https://www.kaggle.com/yasunorim) (Notebooks Expert)
