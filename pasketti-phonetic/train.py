"""Phonetic Track: Wav2Vec2 + CTC fine-tuning for IPA transcription.

Raw PyTorch training loop — compatible with GPU, TPU (torch_xla), and CPU.
No HuggingFace Trainer dependency (avoids TPU PJRT compatibility issues).
"""

import argparse
import gc
import json
import math
import os
import re
import time
import unicodedata
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
import wandb
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

# Prevent tokenizer parallelism deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SAMPLE_RATE = 16000
MAX_DURATION_SEC = 20.0

# IPA normalization (from benchmark score.py)
IPA_VALID_CHARS = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ɑɒæɐɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθ"
    "œɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱɤʌɣɰʍχʎʏʑʐʒʔʕʢʡ"
    "ˈˌːˑ̈̃̊̄̆̋̏̀́̂̌̽̚ʰʷʲˠˤ˞"
    "ɚɝ"
    " "
)


def normalize_ipa(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = "".join(c for c in text if c in IPA_VALID_CHARS or c == " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_transcripts(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def unzip_audio(data_dir: Path):
    for zf_path in sorted(data_dir.glob("*.zip")):
        target = data_dir / zf_path.stem
        if target.exists() and any(target.iterdir()):
            print(f"  Already extracted: {zf_path.name}")
            continue
        print(f"  Extracting: {zf_path.name}")
        with zipfile.ZipFile(zf_path) as zf:
            zf.extractall(data_dir)


def build_vocab(transcripts: list[dict]) -> dict[str, int]:
    char_counts: Counter = Counter()
    for t in transcripts:
        text = normalize_ipa(t["phonetic_text"])
        char_counts.update(text)
    sorted_chars = sorted(char_counts.keys(), key=lambda c: (-char_counts[c], c))
    vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2}
    for c in sorted_chars:
        if c == " ":
            continue
        if c not in vocab:
            vocab[c] = len(vocab)
    print(f"Vocabulary size: {len(vocab)} characters")
    return vocab


def prepare_dataset(
    transcripts: list[dict],
    data_dir: Path,
    processor: Wav2Vec2Processor,
    max_duration: float = MAX_DURATION_SEC,
) -> list[dict]:
    dataset = []
    skipped = 0
    for i, rec in enumerate(transcripts):
        audio_path = data_dir / rec["audio_path"]
        if not audio_path.exists():
            skipped += 1
            continue
        duration = rec.get("audio_duration_sec", 0)
        if duration > max_duration:
            skipped += 1
            continue
        if "U_b8a4e8220e65219b" in str(audio_path):
            skipped += 1
            continue
        text = normalize_ipa(rec["phonetic_text"])
        if not text:
            skipped += 1
            continue
        text = text.replace(" ", "|")
        dataset.append({
            "audio_path": str(audio_path),
            "text": text,
            "duration": duration,
            "utterance_id": rec.get("utterance_id", f"U_{i:06d}"),
        })
    print(f"Prepared {len(dataset)} samples, skipped {skipped}")
    return dataset


class AudioDataset(torch.utils.data.Dataset):
    """Dataset with pre-loaded audio to avoid CPU I/O during training."""

    def __init__(self, data: list[dict], processor, sample_rate=SAMPLE_RATE, preload=False):
        self.data = data
        self.processor = processor
        self.sample_rate = sample_rate
        self.cache = {}
        if preload:
            print(f"  Pre-loading {len(data)} audio files...")
            for i, d in enumerate(data):
                self._load(i)
                if (i + 1) % 1000 == 0:
                    print(f"    {i + 1}/{len(data)} loaded")
            print(f"  Pre-load complete")

    def _load(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        d = self.data[idx]
        speech, sr = sf.read(d["audio_path"])
        if sr != self.sample_rate:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=self.sample_rate)
        inputs = self.processor(speech, sampling_rate=self.sample_rate, return_tensors="pt")
        input_values = inputs.input_values.squeeze(0)
        labels = self.processor.tokenizer(d["text"], return_tensors="pt").input_ids.squeeze(0)
        self.cache[idx] = (input_values, labels)
        return input_values, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._load(idx)


def collate_fn(features, processor=None, sample_rate=SAMPLE_RATE):
    """Collate batch: pad pre-loaded tensors."""
    input_values_list = [f[0] for f in features]
    labels_list = [f[1] for f in features]

    max_input_len = max(iv.shape[0] for iv in input_values_list)
    padded_inputs = torch.zeros(len(input_values_list), max_input_len)
    attention_mask = torch.zeros(len(input_values_list), max_input_len, dtype=torch.long)
    for i, iv in enumerate(input_values_list):
        padded_inputs[i, : iv.shape[0]] = iv
        attention_mask[i, : iv.shape[0]] = 1

    max_label_len = max(l.shape[0] for l in labels_list)
    padded_labels = torch.full((len(labels_list), max_label_len), -100, dtype=torch.long)
    for i, l in enumerate(labels_list):
        padded_labels[i, : l.shape[0]] = l

    return {
        "input_values": padded_inputs,
        "attention_mask": attention_mask,
        "labels": padded_labels,
    }


def get_device():
    """Detect best available device: TPU > GPU > CPU."""
    if os.environ.get("PJRT_DEVICE") == "TPU":
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            print(f"Device: TPU ({device})")
            return device, "tpu"
        except Exception as e:
            print(f"TPU init failed: {e}, falling back")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"Device: GPU ({torch.cuda.get_device_name(0)}, {vram_mb:.0f}MB VRAM)")
        return device, "gpu"
    print("Device: CPU")
    return torch.device("cpu"), "cpu"


def get_safe_batch_size(requested: int, model_name: str, device_type: str) -> int:
    """Auto-reduce batch size based on available VRAM and model size."""
    if device_type != "gpu":
        return requested
    vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
    is_large = any(k in model_name.lower() for k in ["large", "xlsr", "1b", "600m"])
    if is_large:
        max_safe = 2 if vram_mb < 16000 else 4
    else:
        max_safe = 8 if vram_mb < 16000 else 16
    result = min(requested, max_safe)
    if result != requested:
        print(f"  Batch size auto-reduced: {requested} -> {result} (VRAM={vram_mb:.0f}MB)")
    return result


def log_gpu_memory(prefix: str = ""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  [GPU mem{' ' + prefix if prefix else ''}] alloc={alloc:.0f}MB reserved={reserved:.0f}MB")


def compute_cer_batch(pred_ids, label_ids, processor):
    """Compute CER for a batch."""
    from jiwer import cer
    pred_str = processor.batch_decode(pred_ids)
    label_ids_clean = label_ids.clone()
    label_ids_clean[label_ids_clean == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids_clean, group_tokens=False)
    pairs = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
    if not pairs:
        return 1.0
    pred_str, label_str = zip(*pairs)
    return cer(list(label_str), list(pred_str))


def evaluate(model, dataloader, device, device_type, processor):
    """Run evaluation and return average CER."""
    model.eval()
    total_cer = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_values=input_values, attention_mask=attention_mask)
            pred_ids = torch.argmax(outputs.logits, dim=-1)
            batch_cer = compute_cer_batch(pred_ids.cpu(), labels.cpu(), processor)
            total_cer += batch_cer
            total_batches += 1
            if device_type == "tpu":
                import torch_xla.core.xla_model as xm
                xm.mark_step()
    return total_cer / max(total_batches, 1)


def get_cosine_lr(step, warmup_steps, total_steps, base_lr):
    """Cosine schedule with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def dry_run(model_name: str = "facebook/wav2vec2-base"):
    """Quick smoke test: 1 forward+backward pass with dummy data on CPU.
    Catches import errors, dtype mismatches, shape errors before Kaggle push."""
    print("=== DRY RUN: smoke test on CPU ===")
    device = torch.device("cpu")

    vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3, "b": 4, "c": 5}
    vocab_path = Path("/tmp/dry_run_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=SAMPLE_RATE, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    model = Wav2Vec2ForCTC.from_pretrained(
        model_name, ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        mask_time_prob=0.0,
    )
    model.freeze_feature_encoder()
    model.to(device)

    # Dummy batch: 2 samples, 1 second of audio each
    dummy_audio = torch.randn(2, SAMPLE_RATE)
    attention_mask = torch.ones(2, SAMPLE_RATE, dtype=torch.long)
    labels = torch.tensor([[3, 4, 2, 5], [3, 2, 4, 5]], dtype=torch.long)

    # Forward + backward
    outputs = model(input_values=dummy_audio, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()

    from jiwer import cer
    pred_ids = torch.argmax(outputs.logits, dim=-1)
    batch_cer = compute_cer_batch(pred_ids, labels, processor)

    print(f"  Loss: {loss.item():.4f}, CER: {batch_cer:.4f}")
    print("=== DRY RUN PASSED ===")
    vocab_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="model_phonetic")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--wandb_project", type=str, default="drivendata-phonetic-asr")
    parser.add_argument("--memo", type=str, default="local")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device, device_type = get_device()

    # Unzip audio
    print("=== Extracting audio ===")
    unzip_audio(data_dir)

    # Load transcripts
    print("=== Loading transcripts ===")
    transcripts = load_transcripts(data_dir / "train_phon_transcripts.jsonl")
    print(f"Total transcripts: {len(transcripts)}")

    # Build vocabulary
    print("=== Building vocabulary ===")
    vocab = build_vocab(transcripts)
    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, ensure_ascii=False)

    # Create tokenizer and processor
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=SAMPLE_RATE, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Prepare dataset
    print("=== Preparing dataset ===")
    dataset = prepare_dataset(transcripts, data_dir, processor)

    # Train/val split
    np.random.seed(42)
    indices = np.random.permutation(len(dataset))
    split_idx = int(len(dataset) * args.train_split)
    train_data = [dataset[i] for i in indices[:split_idx]]
    val_data = [dataset[i] for i in indices[split_idx:]]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    print("=== Pre-loading audio ===")
    train_dataset = AudioDataset(train_data, processor, preload=True)
    val_dataset = AudioDataset(val_data, processor, preload=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Auto-adjust batch size for GPU safety
    args.batch_size = get_safe_batch_size(args.batch_size, args.model_name, device_type)
    # Auto-adjust gradient accumulation to preserve effective batch size
    effective_target = 64  # reasonable default
    if args.batch_size * args.gradient_accumulation < effective_target:
        args.gradient_accumulation = max(1, effective_target // args.batch_size)
    print(f"  Effective batch: {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")

    # Load model
    print(f"=== Loading model: {args.model_name} ===")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        # Prevent crash on short audio segments
        mask_time_prob=0.0,
    )
    model.freeze_feature_encoder()
    # Gradient checkpointing: enable on GPU (saves ~40% VRAM), disable on TPU (incompatible)
    if device_type == "tpu" and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        print("  Gradient checkpointing: DISABLED (TPU)")
    elif device_type == "gpu":
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: ENABLED (saves ~40% VRAM)")
    model = model.to(device)
    log_gpu_memory("after model load")
    print(f"Model moved to {device}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    # Schedule
    steps_per_epoch = len(train_loader) // args.gradient_accumulation
    total_steps = steps_per_epoch * args.epochs

    # W&B
    run = wandb.init(project=args.wandb_project, name=args.memo, config=vars(args))

    # Training loop
    print(f"=== Training ({args.epochs} epochs, {total_steps} steps) ===")
    global_step = 0
    best_cer = float("inf")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / args.gradient_accumulation
            loss.backward()

            # Mark step every batch to prevent TPU idle timeout
            if device_type == "tpu":
                import torch_xla.core.xla_model as xm
                xm.mark_step()

            if (batch_idx + 1) % args.gradient_accumulation == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if device_type == "tpu":
                    import torch_xla.core.xla_model as xm
                    xm.optimizer_step(optimizer)
                    xm.mark_step()
                else:
                    optimizer.step()

                # LR schedule
                lr = get_cosine_lr(global_step, args.warmup_steps, total_steps, args.lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                optimizer.zero_grad()
                global_step += 1
                epoch_loss += loss.item() * args.gradient_accumulation

                # Logging + periodic GC
                if global_step % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"  Step {global_step}/{total_steps} | Loss: {loss.item() * args.gradient_accumulation:.4f} | LR: {lr:.2e} | {elapsed:.0f}s")
                    wandb.log({"train/loss": loss.item() * args.gradient_accumulation, "train/lr": lr, "train/step": global_step})
                    log_gpu_memory(f"step {global_step}")
                    gc.collect()
                    if device_type == "gpu":
                        torch.cuda.empty_cache()

                # Eval + checkpoint
                if global_step % args.eval_steps == 0:
                    cer_score = evaluate(model, val_loader, device, device_type, processor)
                    print(f"  [Eval] Step {global_step} | CER: {cer_score:.4f}")
                    wandb.log({"eval/cer": cer_score, "eval/step": global_step})

                    # Save checkpoint
                    ckpt_path = checkpoint_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(str(ckpt_path))
                    processor.save_pretrained(str(ckpt_path))
                    print(f"  Saved checkpoint: {ckpt_path}")

                    if cer_score < best_cer:
                        best_cer = cer_score
                        best_path = checkpoint_dir / "best"
                        model.save_pretrained(str(best_path))
                        processor.save_pretrained(str(best_path))
                        print(f"  New best CER: {best_cer:.4f}")

                    # Cleanup old checkpoints (keep best + latest 3)
                    ckpts = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
                    for old in ckpts[:-3]:
                        import shutil
                        shutil.rmtree(old)

                    model.train()

        avg_loss = epoch_loss / max(steps_per_epoch, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} | Avg Loss: {avg_loss:.4f} | Time: {time.time() - t0:.0f}s")

    # Final evaluation
    print("=== Final evaluation ===")
    final_cer = evaluate(model, val_loader, device, device_type, processor)
    print(f"Final CER: {final_cer:.4f} (Best: {best_cer:.4f})")
    wandb.log({"final_cer": final_cer, "best_cer": best_cer})

    # Save final model (use best if available)
    print("=== Saving model ===")
    model_save_dir = output_dir / "final_model"
    best_path = checkpoint_dir / "best"
    if best_path.exists():
        import shutil
        shutil.copytree(best_path, model_save_dir, dirs_exist_ok=True)
        print(f"Saved best model (CER: {best_cer:.4f}) to {model_save_dir}")
    else:
        model.save_pretrained(str(model_save_dir))
        processor.save_pretrained(str(model_save_dir))
        print(f"Saved final model to {model_save_dir}")

    # Save vocab
    with open(model_save_dir / "vocab.json", "w") as f:
        json.dump(vocab, f, ensure_ascii=False)

    total_size = sum(f.stat().st_size for f in model_save_dir.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"Model size: {total_size:.1f} MB")
    run.finish()


if __name__ == "__main__":
    import sys
    if "--dry_run" in sys.argv:
        dry_run()
    else:
        main()
