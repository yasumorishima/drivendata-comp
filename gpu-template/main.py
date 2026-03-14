"""GPU competition inference template for DrivenData code submission.

This file runs inside the DrivenData runtime container.
Network is blocked — all model weights must be in the ZIP.
"""

import json
from pathlib import Path

MODEL_DIR = Path(__file__).parent.resolve() / "model"
DATA_DIR = Path("/code_execution/data")
SUBMISSION_DIR = Path("/code_execution/submission")


def load_manifest(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main():
    # --- Load model ---
    # TODO: load model from MODEL_DIR

    # --- Load data ---
    manifest = load_manifest(DATA_DIR / "utterance_metadata.jsonl")

    # --- Inference ---
    results = {}
    # TODO: implement inference

    # --- Write submission ---
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUBMISSION_DIR / "submission.jsonl"
    with open(output_path, "w") as f:
        for uid, pred in sorted(results.items()):
            f.write(json.dumps({"utterance_id": uid, "prediction": pred}) + "\n")

    print(f"Submission written: {output_path} ({len(results)} predictions)")


if __name__ == "__main__":
    main()
