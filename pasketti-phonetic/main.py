"""Phonetic Track inference script for DrivenData submission.

This file runs inside the DrivenData runtime container.
Reads audio from data/, writes predictions to submission/submission.jsonl.

Runtime: Python 3.11, CUDA 12.6, transformers>=4.52.4, torch>=2.9.0
"""

import json
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

SAMPLE_RATE = 16000
BATCH_SIZE = 8
MODEL_DIR = Path(__file__).parent.resolve() / "model"

# Paths inside DrivenData runtime container
DATA_DIR = Path("/code_execution/data")
SUBMISSION_DIR = Path("/code_execution/submission")


def load_manifest(path: Path) -> list[dict]:
    """Load utterance metadata JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model and processor
    print(f"Loading model from {MODEL_DIR}")
    processor = Wav2Vec2Processor.from_pretrained(str(MODEL_DIR))
    model = Wav2Vec2ForCTC.from_pretrained(str(MODEL_DIR)).to(device)
    model.eval()

    # Load manifest
    manifest_path = DATA_DIR / "utterance_metadata.jsonl"
    manifest = load_manifest(manifest_path)
    print(f"Total utterances: {len(manifest)}")

    # Sort by duration (longest first for efficient batching)
    manifest.sort(key=lambda x: x.get("audio_duration_sec", 0), reverse=True)

    # Load submission format for utterance IDs
    format_path = DATA_DIR / "submission_format.jsonl"
    expected_ids = set()
    if format_path.exists():
        with open(format_path) as f:
            for line in f:
                rec = json.loads(line)
                expected_ids.add(rec["utterance_id"])

    # Batch inference
    results = {}
    for batch_start in range(0, len(manifest), BATCH_SIZE):
        batch = manifest[batch_start : batch_start + BATCH_SIZE]

        speeches = []
        utterance_ids = []
        for rec in batch:
            audio_path = DATA_DIR / rec["audio_filepath"]
            speech, sr = sf.read(str(audio_path))
            if sr != SAMPLE_RATE:
                speech = librosa.resample(speech, orig_sr=sr, target_sr=SAMPLE_RATE)
            speeches.append(speech)
            utterance_ids.append(rec["utterance_id"])

        # Process
        inputs = processor(
            speeches,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)
        pred_texts = processor.batch_decode(pred_ids)

        for uid, text in zip(utterance_ids, pred_texts):
            # Replace word boundary token with space
            text = text.replace("|", " ").strip()
            results[uid] = text

        if (batch_start // BATCH_SIZE) % 50 == 0:
            print(f"  Processed {batch_start + len(batch)}/{len(manifest)}")

    # Write submission
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUBMISSION_DIR / "submission.jsonl"

    with open(output_path, "w") as f:
        # If we have a submission format, follow its order
        if expected_ids:
            for uid in sorted(expected_ids):
                text = results.get(uid, "")
                f.write(json.dumps({"utterance_id": uid, "phonetic_text": text}) + "\n")
        else:
            for uid, text in sorted(results.items()):
                f.write(json.dumps({"utterance_id": uid, "phonetic_text": text}) + "\n")

    print(f"Submission written: {output_path} ({len(results)} predictions)")


if __name__ == "__main__":
    main()
