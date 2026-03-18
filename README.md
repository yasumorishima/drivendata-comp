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

**Phonetic Track (in progress)**
- Base model: `facebook/wav2vec2-base` / `facebook/wav2vec2-large-xlsr-53` with CTC head
- IPA vocabulary built from training transcripts (100+ IPA characters)
- Cosine LR decay + gradient accumulation
- Checkpoint resume: session切れでも途中から再開可能

**Word Track (planned)**
- Base model: NVIDIA Parakeet TDT 0.6B
- Noise augmentation using provided classroom noise samples

---

## Pipeline Overview

### Experiment Iteration (Colab GPU)

```
[Local PC]                  [RPi5 (xrdp)]               [Google Colab (Free)]
Claude Code                 Remote Desktop session       Experiment Runner v4
  ↓ Write config to Drive     ↓ RDP for OAuth/setup        ↓ Monitor Drive for configs
  ↓                          ↓ Session persists on DC      ↓ Download data (GH Artifact → local)
Google Drive (for Desktop) ←――――――――――――――――――→ Google Drive (mount)
  EXP/config/child-exp005.yaml                          Detect config → setup_data.py → train.py
  EXP/output/child-exp005/result.json                   Results saved to Drive
```

- Training data is downloaded to Colab local storage (ephemeral, not Drive) to save quota
- RPi5 xrdp allows manual OAuth/2FA approval; RDP session persists after disconnect
- CDP keepalive (systemd) executes JS via Chrome DevTools Protocol every 20 min to prevent Colab idle timeout
- Claude Code session can be closed while GPU training runs
- Same methodology as [kaggle-competitions](https://github.com/yasumorishima/kaggle-competitions#-experiment-management-exp--child-exp)

### Final Training & Submission (GitHub Actions + Kaggle GPU)

```
Download Data (Playwright) → Kaggle P100 Train → GitHub Release → Package ZIP → Submit
```

- Best config from Colab experiments → full training on Kaggle GPU
- GPU→CPU auto-fallback on quota/OOM/CUDA errors
- `COMPLETE_EMPTY` (no output) treated as failure

### Drive Structure

```
Google Drive/kaggle/
├── runner/
│   └── experiment_runner_v4.ipynb  # v4.1: multi-comp, real-time logging, GPU check, heartbeat
└── pasketti/
    ├── requirements.txt         # ASR deps (librosa, soundfile, jiwer)
    ├── setup_data.py            # Downloads data from GH Artifact to Colab local
    └── EXP/
        ├── requirements.txt     # Copy for runner discovery
        └── EXP001/
            ├── train.py         # Wav2Vec2 CTC fine-tuning
            ├── config/
            │   ├── child-exp000.yaml   # wav2vec2-base (batch=16, grad_accum=4)
            │   └── child-exp001.yaml   # wav2vec2-large-xlsr-53 (batch=2, grad_accum=32)
            └── output/          # Results (result.json, train.log, checkpoints)
```

## Workflows

| Workflow | Purpose | Trigger |
|---|---|---|
| **Check Competitions** | List active DrivenData competitions | `workflow_dispatch` |
| **Download Competition Data** | Playwright auto-download → Artifact | `workflow_dispatch` |
| **DrivenData Train & Validate** | CSV submission: train → validate → submission.csv | `workflow_dispatch` |
| **DrivenData GPU Train (Kaggle)** | Code submission: Kaggle P100 train → Release | `workflow_dispatch` |
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

# 3. Package for submission
gh workflow run "Package DrivenData Submission" \
  -f competition_dir=pasketti-phonetic \
  -f model_release_tag=phonetic-model-v3 \
  -f memo="baseline submission"
```

## Training Flows

3つの学習環境を使い分ける。GPU/TPU枠は別カウントなので、一方が切れてももう一方で学習可能。

| Flow | Accelerator | Use Case | Quota |
|---|---|---|---|
| **Kaggle GPU** | P100 | 本番学習 | 週30h（土曜リセット） |
| **Kaggle TPU** | TPU v3-8 (128GB HBM) | GPU枠切れ時の代替 | 週30h（GPU枠とは別） |
| **Colab GPU** | T4 (RPi5経由) | 実験イテレーション | 1日数時間（12-24hリセット） |

```
学習したい
  ├── 実験イテレーション → Colab GPU（EXP + child-exp）
  ├── 本番学習
  │   ├── Kaggle GPU枠あり → DrivenData GPU Train (Kaggle)
  │   ├── GPU枠切れ、TPU枠あり → DrivenData TPU Train (Kaggle)
  │   └── 両方切れ → 翌日待ち
```

### Checkpoint Resume（セッション切れ対策）

- 200ステップごとにチェックポイント保存
- 保存直後に `os.sync()` で Google Drive に即flush
- 次回起動時、最新チェックポイントから自動再開
- `save_total_limit=3`（直近3世代保持）

## Tech Stack

| Component | Technology |
|---|---|
| Training | PyTorch + HuggingFace Transformers |
| ASR Model | Wav2Vec2 (CTC) / Parakeet TDT (planned) |
| Accelerator | Colab T4 (iteration) / Kaggle P100 (GPU) / Kaggle TPU v3-8 |
| CI/CD | GitHub Actions |
| Experiment Tracking | W&B (offline sync) |
| Notifications | Discord Webhook |
| Data Download | Playwright (headless browser) |
| Session Keepalive | RPi5 + xrdp + Chromium CDP (Chrome DevTools Protocol) |

## Project Structure

```
drivendata-comp/
├── .github/workflows/        # GitHub Actions pipelines
├── template/                 # CSV submission template
├── gpu-template/             # Code submission template (GPU)
├── pasketti-phonetic/        # Phonetic Track: Wav2Vec2 CTC
│   ├── train.py              # Training script (runs on Kaggle GPU)
│   ├── main.py               # Inference script (DrivenData runtime)
│   ├── generate_notebook.py  # Embeds train.py into Kaggle notebook
│   └── kernel-metadata.json
├── pasketti-word/            # Word Track (TBD)
├── scripts/                  # Utility scripts
└── DRIVENDATA_MEMO.md        # Internal operation notes
```

## Roadmap

- [x] Pipeline: Download → Kaggle GPU Train → Release (GPU→CPU fallback)
- [x] EXP + child-exp experiment iteration via Colab + Google Drive
- [x] Runner v4.1: real-time logging, GPU enforcement, GH_TOKEN Drive fallback, heartbeat
- [x] CDP keepalive: Chrome DevTools Protocol via systemd (replaces xdotool)
- [x] Kaggle TPU v3-8 training workflow (GPU枠とは別枠で学習可能)
- [x] Checkpoint resume with Drive flush (200 steps保存、os.sync()で即flush)
- [ ] Phonetic child-exp000 (wav2vec2-base CTC baseline) — TPU v3-8で実行中 (3/19)
- [ ] Phonetic child-exp001 (wav2vec2-large-xlsr-53) — OOM修正済み、exp000後に実行
- [ ] Package Submission → first DrivenData submission (CER score)
- [ ] Phonetic improvements: data augmentation, pyctcdecode LM
- [ ] Word Track: Parakeet TDT 0.6B (17.3GB audio data)

## Profile

- DrivenData: [Ymori](https://www.drivendata.org/users/Ymori/)
- Kaggle: [yasunorim](https://www.kaggle.com/yasunorim) (Notebooks Expert)
