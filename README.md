# drivendata-comp

DrivenData competition pipeline — automated training, packaging, and submission via GitHub Actions + Kaggle GPU.

---

## Pipeline Overview

```
1. Download Competition Data  →  Artifact (Playwright auto-DL)
2. GPU Train (Kaggle P100)    →  Model weights → GitHub Release
3. Package Submission          →  main.py + model → submission.zip
4. Manual submit on DrivenData
```

### GPU → CPU Fallback

GPU session limit or quota exhausted → immediate CPU fallback (Discord notification).
GPU-related error during execution → automatic CPU retry.

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

## Active Competitions

| Competition | Track | Metric | Deadline |
|---|---|---|---|
| [On Top of Pasketti](https://www.drivendata.org/competitions/309/) | Phonetic (IPA CER) | CER | 2026-04-06 |
| [On Top of Pasketti](https://www.drivendata.org/competitions/308/) | Word (WER) | WER | 2026-04-06 |

## Project Structure

```
drivendata-comp/
├── .github/workflows/     # GitHub Actions pipelines
├── template/              # CSV submission template
├── gpu-template/          # Code submission template (GPU)
├── pasketti-phonetic/     # Phonetic Track: Wav2Vec2 CTC
│   ├── train.py           # Training script (runs on Kaggle GPU)
│   ├── main.py            # Inference script (DrivenData runtime)
│   ├── generate_notebook.py  # Embeds train.py into Kaggle notebook
│   └── kernel-metadata.json
├── pasketti-word/         # Word Track (TBD)
├── scripts/               # Utility scripts
└── DRIVENDATA_MEMO.md     # Internal operation notes
```

## Roadmap

- [ ] Phonetic v1 結果確認 → 全自動パイプラインテスト（Push → Poll → Release）
- [ ] Package Submission → DrivenData初回提出（CERスコア確認）
- [ ] Phonetic改善: wav2vec2-large-xlsr-53, data augmentation, LM decode
- [ ] Word Track着手（Parakeet TDT 0.6B, 17.3GB audio data）

## Profile

- DrivenData: [Ymori](https://www.drivendata.org/users/Ymori/)
- Kaggle: [yasunorim](https://www.kaggle.com/yasunorim) (Notebooks Expert)
