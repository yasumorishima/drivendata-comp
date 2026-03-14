"""Word Track inference script for DrivenData submission.

This file runs inside the DrivenData runtime container.
Reads audio from data/, writes predictions to submission/submission.jsonl.

Runtime: Python 3.11, CUDA 12.6, nemo_toolkit[asr]>=2.5.0, torch>=2.9.0
"""

import json
from pathlib import Path

import torch

SAMPLE_RATE = 16000
BATCH_SIZE = 4
MODEL_PATH = Path(__file__).parent.resolve() / "model" / "final_model.nemo"

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


def patch_transcribe_lhotse(model):
    """Disable lhotse in NeMo transcribe dataloader (PyTorch Sampler API incompatibility)."""
    import types

    original_transcribe = model.transcribe

    def patched_transcribe(self, *args, **kwargs):
        kwargs.setdefault("num_workers", 0)
        return original_transcribe(*args, **kwargs)

    model.transcribe = types.MethodType(patched_transcribe, model)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load NeMo model
    print(f"Loading model from {MODEL_PATH}")
    import nemo.collections.asr as nemo_asr

    asr_model = nemo_asr.models.ASRModel.restore_from(str(MODEL_PATH), map_location=device)
    asr_model.eval()

    # Enable adapter if present
    if hasattr(asr_model, "set_enabled_adapters"):
        asr_model.set_enabled_adapters(["linear_adapter"])

    patch_transcribe_lhotse(asr_model)

    # Load manifest
    manifest_path = DATA_DIR / "utterance_metadata.jsonl"
    manifest = load_manifest(manifest_path)
    print(f"Total utterances: {len(manifest)}")

    # Sort by duration (longest first)
    manifest.sort(key=lambda x: x.get("audio_duration_sec", 0), reverse=True)

    # Load submission format
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
        audio_paths = [str(DATA_DIR / rec["audio_filepath"]) for rec in batch]
        utterance_ids = [rec["utterance_id"] for rec in batch]

        transcriptions = asr_model.transcribe(audio_paths, batch_size=BATCH_SIZE)

        # NeMo transcribe returns list of strings or list of Hypothesis
        if isinstance(transcriptions, tuple):
            transcriptions = transcriptions[0]

        for uid, text in zip(utterance_ids, transcriptions):
            if hasattr(text, "text"):
                text = text.text
            results[uid] = str(text).strip()

        if (batch_start // BATCH_SIZE) % 50 == 0:
            print(f"  Processed {batch_start + len(batch)}/{len(manifest)}")

    # Write submission
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUBMISSION_DIR / "submission.jsonl"

    with open(output_path, "w") as f:
        if expected_ids:
            for uid in sorted(expected_ids):
                text = results.get(uid, "")
                f.write(json.dumps({"utterance_id": uid, "orthographic_text": text}) + "\n")
        else:
            for uid, text in sorted(results.items()):
                f.write(json.dumps({"utterance_id": uid, "orthographic_text": text}) + "\n")

    print(f"Submission written: {output_path} ({len(results)} predictions)")


if __name__ == "__main__":
    main()
