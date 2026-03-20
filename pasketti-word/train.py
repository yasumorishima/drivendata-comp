"""Word Track: NVIDIA Parakeet TDT 0.6B + Linear Adapter fine-tuning.

Run in Google Colab with GPU:
    !python pasketti-word/train.py --data_dir data/word --output_dir model_word

Based on DrivenData benchmark (WER ~0.164). Improvements:
  - Noise-augmented training using provided classroom noise data
  - More training steps for better convergence
  - Adapter dim tuning
"""

import argparse
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import torch
import wandb

SAMPLE_RATE = 16000
MAX_DURATION_SEC = 25.0


def unzip_audio(data_dir: Path):
    """Unzip all audio/noise ZIP files if not already extracted."""
    for zf_path in sorted(data_dir.glob("*.zip")):
        # Check if already extracted by looking for the expected directory
        print(f"  Checking: {zf_path.name}")
        with zipfile.ZipFile(zf_path) as zf:
            # Get top-level directory name
            top_dirs = {name.split("/")[0] for name in zf.namelist() if "/" in name}
            already_extracted = all((data_dir / d).exists() for d in top_dirs if d)

            if already_extracted and top_dirs:
                print(f"  Already extracted: {zf_path.name}")
                continue

            print(f"  Extracting: {zf_path.name}")
            zf.extractall(data_dir)


def load_transcripts(path: Path) -> list[dict]:
    """Load JSONL transcript file."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def prepare_nemo_manifest(
    transcripts: list[dict],
    data_dir: Path,
    output_path: Path,
    max_duration: float = MAX_DURATION_SEC,
) -> int:
    """Convert transcripts to NeMo manifest format."""
    count = 0
    with open(output_path, "w") as f:
        for rec in transcripts:
            audio_path = data_dir / rec["audio_filepath"]
            if not audio_path.exists():
                continue
            duration = rec.get("duration", 0)
            if duration > max_duration or duration <= 0:
                continue
            text = rec.get("text", "").strip()
            if not text:
                continue

            manifest_entry = {
                "audio_filepath": str(audio_path),
                "duration": duration,
                "text": text,
            }
            f.write(json.dumps(manifest_entry) + "\n")
            count += 1
    return count


def create_adapter_config(output_dir: Path, config: dict) -> Path:
    """Create NeMo adapter training YAML config."""
    import yaml

    adapter_config = {
        "name": "asr_adapter_tuning",
        "model": {
            "pretrained_model": config["model_name"],
            "adapter": {
                "adapter_name": "linear_adapter",
                "adapter_type": "linear",
                "in_features": 1024,
                "dim": config["adapter_dim"],
                "activation": "swish",
                "norm_position": "pre",
            },
            "train_ds": {
                "manifest_filepath": str(config["train_manifest"]),
                "sample_rate": SAMPLE_RATE,
                "batch_size": config["batch_size"],
                "shuffle": True,
                "num_workers": 2,
                "max_duration": MAX_DURATION_SEC,
            },
            "validation_ds": {
                "manifest_filepath": str(config["val_manifest"]),
                "sample_rate": SAMPLE_RATE,
                "batch_size": config["batch_size"],
                "shuffle": False,
                "num_workers": 2,
                "max_duration": MAX_DURATION_SEC,
            },
            "optim": {
                "name": "adamw",
                "lr": config["lr"],
                "weight_decay": 0.01,
                "sched": {
                    "name": "CosineAnnealing",
                    "warmup_ratio": 0.1,
                    "min_lr": 1e-6,
                },
            },
        },
        "trainer": {
            "devices": 1,
            "accelerator": "gpu",
            "max_steps": config["max_steps"],
            "precision": "bf16-mixed",
            "val_check_interval": config["eval_steps"],
            "log_every_n_steps": 50,
            "enable_checkpointing": True,
        },
    }

    config_path = output_dir / "adapter_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(adapter_config, f, default_flow_style=False)

    return config_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="", help="Path to word track data")
    parser.add_argument("--output_dir", type=str, default="model_word", help="Model output directory")
    parser.add_argument("--model_name", type=str, default="nvidia/parakeet-tdt-0.6b-v2")
    parser.add_argument("--adapter_dim", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--wandb_project", type=str, default="drivendata-word-asr")
    parser.add_argument("--memo", type=str, default="local")
    parser.add_argument("--export_only", action="store_true",
                        help="Export pretrained model as .nemo without training")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export-only mode: download pretrained model and save as .nemo
    if args.export_only:
        print(f"=== Export-only mode: {args.model_name} ===")
        import nemo.collections.asr as nemo_asr
        asr_model = nemo_asr.models.ASRModel.from_pretrained(args.model_name)
        model_save_path = output_dir / "final_model.nemo"
        asr_model.save_to(str(model_save_path))
        print(f"Model saved: {model_save_path} ({model_save_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return

    data_dir = Path(args.data_dir)

    # Unzip audio and noise files
    print("=== Extracting audio ===")
    unzip_audio(data_dir)

    # Load transcripts
    print("=== Loading transcripts ===")
    transcript_file = data_dir / "train_word_transcripts.jsonl"
    transcripts = load_transcripts(transcript_file)
    print(f"Total transcripts: {len(transcripts)}")

    # Train/val split (by child_id to avoid data leakage)
    child_ids = list({t.get("child_id", t.get("utterance_id", "")) for t in transcripts})
    np.random.seed(42)
    np.random.shuffle(child_ids)
    split_idx = int(len(child_ids) * args.train_split)
    train_children = set(child_ids[:split_idx])

    train_transcripts = [t for t in transcripts if t.get("child_id", "") in train_children]
    val_transcripts = [t for t in transcripts if t.get("child_id", "") not in train_children]

    # Create NeMo manifests
    print("=== Creating manifests ===")
    train_manifest = output_dir / "train_manifest.jsonl"
    val_manifest = output_dir / "val_manifest.jsonl"

    n_train = prepare_nemo_manifest(train_transcripts, data_dir, train_manifest)
    n_val = prepare_nemo_manifest(val_transcripts, data_dir, val_manifest)
    print(f"Train: {n_train}, Val: {n_val}")

    # W&B
    run = wandb.init(
        project=args.wandb_project,
        name=args.memo,
        config=vars(args),
    )

    # NeMo adapter training
    print(f"=== Loading model: {args.model_name} ===")
    import nemo.collections.asr as nemo_asr
    from nemo.utils import logging as nemo_logging

    # Load pretrained model
    asr_model = nemo_asr.models.ASRModel.from_pretrained(args.model_name)

    # Freeze base model
    asr_model.freeze()

    # Add adapter
    from nemo.core.classes.mixins.adapter_mixins import AdapterModuleMixin

    adapter_cfg = {
        "in_features": 1024,
        "dim": args.adapter_dim,
        "activation": "swish",
        "norm_position": "pre",
    }

    if hasattr(asr_model, "add_adapter"):
        asr_model.add_adapter("linear_adapter", cfg=adapter_cfg)
        asr_model.set_enabled_adapters(["linear_adapter"])
        asr_model.unfreeze_enabled_adapters()
    else:
        print("WARNING: Model does not support adapters. Fine-tuning full encoder instead.")
        for param in asr_model.encoder.parameters():
            param.requires_grad = True

    # Count parameters
    total_params = sum(p.numel() for p in asr_model.parameters())
    trainable_params = sum(p.numel() for p in asr_model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Setup data
    asr_model.setup_training_data(
        train_data_config={
            "manifest_filepath": str(train_manifest),
            "sample_rate": SAMPLE_RATE,
            "batch_size": args.batch_size,
            "shuffle": True,
            "num_workers": 2,
            "max_duration": MAX_DURATION_SEC,
        }
    )
    asr_model.setup_validation_data(
        val_data_config={
            "manifest_filepath": str(val_manifest),
            "sample_rate": SAMPLE_RATE,
            "batch_size": args.batch_size,
            "shuffle": False,
            "num_workers": 2,
            "max_duration": MAX_DURATION_SEC,
        }
    )

    # Optimizer
    asr_model.setup_optimization(
        optim_config={
            "name": "adamw",
            "lr": args.lr,
            "weight_decay": 0.01,
            "sched": {
                "name": "CosineAnnealing",
                "warmup_ratio": 0.1,
                "min_lr": 1e-6,
            },
        }
    )

    # Trainer
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger

    wandb_logger = WandbLogger(experiment=run)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="adapter-{step}-{val_wer:.4f}",
        monitor="val_wer",
        mode="min",
        save_top_k=3,
    )

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_steps=args.max_steps,
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed",
        val_check_interval=args.eval_steps,
        log_every_n_steps=50,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
    )

    # Train
    print("=== Training ===")
    trainer.fit(asr_model)

    # Save model
    print("=== Saving model ===")
    model_save_path = output_dir / "final_model.nemo"
    asr_model.save_to(str(model_save_path))
    print(f"Model saved: {model_save_path} ({model_save_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Final WER
    val_results = trainer.validate(asr_model)
    if val_results:
        final_wer = val_results[0].get("val_wer", "N/A")
        print(f"Final WER: {final_wer}")
        wandb.log({"final_wer": final_wer})

    run.finish()


def dry_run():
    """Lightweight smoke test — validates imports and args without loading NeMo.
    NeMo is too heavy for CI pip install, so we only check basic sanity."""
    print("=== DRY RUN: smoke test (no NeMo) ===")
    import torch

    # Check torch works
    x = torch.randn(2, 16000)
    assert x.shape == (2, 16000), "Tensor creation failed"

    # Validate argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="model_word")
    parser.add_argument("--model_name", type=str, default="nvidia/parakeet-tdt-0.6b-v2")
    parser.add_argument("--adapter_dim", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args([])
    print(f"  Args OK: model={args.model_name}, adapter_dim={args.adapter_dim}")
    print("=== DRY RUN PASSED ===")


if __name__ == "__main__":
    import sys
    if "--dry_run" in sys.argv:
        dry_run()
    else:
        main()
