# drivendata-comp

[DrivenData](https://www.drivendata.org/) competition pipeline — automated training, packaging, and submission via GitHub Actions + Kaggle GPU.

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
- Base model: `facebook/wav2vec2-base` with CTC head
- IPA vocabulary built from training transcripts (100+ IPA characters)
- SpecAugment + cosine LR decay + gradient accumulation
- Training on Kaggle P100 GPU

**Word Track (planned)**
- Base model: NVIDIA Parakeet TDT 0.6B
- Noise augmentation using provided classroom noise samples

---

## Pipeline Overview

```
┌─────────────────────────┐
│ 1. Download Data        │  GitHub Actions + Playwright
│    → Artifact           │  (auto-login, auto-download)
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│ 2. GPU Train            │  Kaggle P100 GPU
│    train.py (base64)    │  (GPU→CPU auto-fallback)
│    → model weights      │
│    → GitHub Release     │
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│ 3. Package Submission   │  GitHub Actions
│    main.py + model/     │  → submission.zip
│    → Artifact           │
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│ 4. Manual Submit        │  Upload ZIP on DrivenData
└─────────────────────────┘
```

### GPU → CPU Fallback

- GPU kernel failure (quota, OOM, CUDA error, etc.) → automatic CPU retry + Discord notification
- Polling waits for `RUNNING`/`QUEUED` transition before checking completion (stale status防止)
- `COMPLETE_EMPTY` (no output) is treated as failure, not success

## Workflows

| Workflow | Purpose | Trigger |
|---|---|---|
| **Check Competitions** | List active DrivenData competitions | `workflow_dispatch` |
| **Download Competition Data** | Playwright auto-download → Artifact | `workflow_dispatch` |
| **DrivenData Train & Validate** | CSV submission: train → validate → submission.csv | `workflow_dispatch` |
| **DrivenData GPU Train (Kaggle)** | Code submission: Kaggle GPU train → Release | `workflow_dispatch` |
| **Package DrivenData Submission** | Code submission: Release model + main.py → ZIP | `workflow_dispatch` |

## Quick Start

```bash
# 1. Download data
gh workflow run "Download Competition Data" \
  --repo yasumorishima/drivendata-comp \
  -f memo="initial download"

# 2. Train on Kaggle GPU
gh workflow run "DrivenData GPU Train (Kaggle)" \
  --repo yasumorishima/drivendata-comp \
  -f competition_dir=pasketti-phonetic \
  -f model_release_tag=phonetic-model-v1 \
  -f memo="v1: wav2vec2-base baseline"

# 3. Package for submission
gh workflow run "Package DrivenData Submission" \
  --repo yasumorishima/drivendata-comp \
  -f competition_dir=pasketti-phonetic \
  -f model_release_tag=phonetic-model-v1 \
  -f memo="v1: baseline submission"
```

## Tech Stack

| Component | Technology |
|---|---|
| Training | PyTorch + HuggingFace Transformers |
| ASR Model | Wav2Vec2 (CTC) / Parakeet TDT (planned) |
| GPU | Kaggle P100 (free tier) |
| CI/CD | GitHub Actions |
| Experiment Tracking | W&B (offline sync from Kaggle) |
| Notifications | Discord Webhook |
| Data Download | Playwright (headless browser) |

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

- [ ] Phonetic v1 results → end-to-end pipeline test (Push → Poll → Release)
- [ ] Package Submission → first DrivenData submission (CER score)
- [ ] Phonetic improvements: wav2vec2-large-xlsr-53, data augmentation, LM decode
- [ ] Word Track: Parakeet TDT 0.6B (17.3GB audio data)

## Profile

- DrivenData: [Ymori](https://www.drivendata.org/users/Ymori/)
- Kaggle: [yasunorim](https://www.kaggle.com/yasunorim) (Notebooks Expert)
