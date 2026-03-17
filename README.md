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
- Base model: `facebook/wav2vec2-base` with CTC head
- IPA vocabulary built from training transcripts (100+ IPA characters)
- SpecAugment + cosine LR decay + gradient accumulation

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
│   └── experiment_runner.ipynb  # v3.1: multi-comp, real-time logging, GPU check
└── pasketti/
    ├── requirements.txt         # ASR deps (Colab未搭載のもののみ: librosa, soundfile, jiwer)
    ├── setup_data.py            # Downloads data from GH Artifact to Colab local
    ├── EXP_SUMMARY.md
    ├── CLAUDE_COMP.md
    └── EXP/EXP001/
        ├── train.py             # Wav2Vec2 CTC fine-tuning
        ├── config/
        │   ├── child-exp000.yaml   # wav2vec2-base baseline
        │   └── child-exp001.yaml   # wav2vec2-large-xlsr-53
        └── output/                 # Results saved here (result.json, train.log)
```

## Workflows

| Workflow | Purpose | Trigger |
|---|---|---|
| **Check Competitions** | List active DrivenData competitions | `workflow_dispatch` |
| **Download Competition Data** | Playwright auto-download → Artifact | `workflow_dispatch` |
| **DrivenData Train & Validate** | CSV submission: train → validate → submission.csv | `workflow_dispatch` |
| **DrivenData GPU Train (Kaggle)** | Code submission: Kaggle P100 train → Release | `workflow_dispatch` |
| **Package DrivenData Submission** | Code submission: Release model + main.py → ZIP | `workflow_dispatch` |

## Quick Start

```bash
# 1. Download data
gh workflow run "Download Competition Data" \
  -f memo="initial download"

# 2. Train on Kaggle GPU (after Colab experiments confirm best config)
gh workflow run "DrivenData GPU Train (Kaggle)" \
  -f competition_dir=pasketti-phonetic \
  -f model_release_tag=phonetic-model-v3 \
  -f memo="wav2vec2-base CTC baseline"

# 3. Package for submission
gh workflow run "Package DrivenData Submission" \
  -f competition_dir=pasketti-phonetic \
  -f model_release_tag=phonetic-model-v3 \
  -f memo="baseline submission"
```

## Tech Stack

| Component | Technology |
|---|---|
| Training | PyTorch + HuggingFace Transformers |
| ASR Model | Wav2Vec2 (CTC) / Parakeet TDT (planned) |
| GPU | Google Colab T4 (iteration) / Kaggle P100 (final) |
| CI/CD | GitHub Actions |
| Experiment Tracking | W&B (offline sync) |
| Notifications | Discord Webhook |
| Data Download | Playwright (headless browser) |
| Session Keepalive | RPi5 + xrdp + Chromium |

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
- [x] Runner v3.1: real-time logging, GPU enforcement, GH_TOKEN Drive fallback
- [ ] Phonetic child-exp000 (wav2vec2-base CTC baseline) — T4 GPU学習中
- [ ] Package Submission → first DrivenData submission (CER score)
- [ ] Phonetic improvements: wav2vec2-large-xlsr-53, data augmentation, LM decode
- [ ] Word Track: Parakeet TDT 0.6B (17.3GB audio data)

## Profile

- DrivenData: [Ymori](https://www.drivendata.org/users/Ymori/)
- Kaggle: [yasunorim](https://www.kaggle.com/yasunorim) (Notebooks Expert)
