"""Phonetic Track: Wav2Vec2 + CTC fine-tuning for IPA transcription.

Run in Google Colab with GPU:
    !python pasketti-phonetic/train.py --data_dir data/phonetic --output_dir model_phonetic

Based on DrivenData benchmark (CER ~0.33). Improvements:
  - wav2vec2-large-xlsr-53 option for multilingual robustness
  - SpecAugment for data augmentation
  - Learning rate warmup + cosine decay
  - Gradient accumulation for effective larger batch
"""

import argparse
import json
import os
import re
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
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

SAMPLE_RATE = 16000
MAX_DURATION_SEC = 25.0

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
    """Normalize IPA text for consistent training."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    # Remove non-IPA characters
    text = "".join(c for c in text if c in IPA_VALID_CHARS or c == " ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_transcripts(path: Path) -> list[dict]:
    """Load JSONL transcript file."""
    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)
    return records


def unzip_audio(data_dir: Path):
    """Unzip audio files if not already extracted."""
    for zf_path in sorted(data_dir.glob("*.zip")):
        target = data_dir / zf_path.stem
        if target.exists() and any(target.iterdir()):
            print(f"  Already extracted: {zf_path.name}")
            continue
        print(f"  Extracting: {zf_path.name}")
        with zipfile.ZipFile(zf_path) as zf:
            zf.extractall(data_dir)


def build_vocab(transcripts: list[dict]) -> dict[str, int]:
    """Build IPA character vocabulary from training transcripts."""
    char_counts: Counter = Counter()
    for t in transcripts:
        text = normalize_ipa(t["phonetic_text"])
        char_counts.update(text)

    # Sort by frequency (descending), then alphabetically for ties
    sorted_chars = sorted(char_counts.keys(), key=lambda c: (-char_counts[c], c))

    vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2}  # | = word boundary
    for i, c in enumerate(sorted_chars):
        if c == " ":
            continue  # space maps to | (word boundary)
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
    """Load audio and tokenize transcripts."""
    dataset = []
    skipped = 0

    for i, rec in enumerate(transcripts):
        audio_path = data_dir / rec["audio_filepath"]
        if not audio_path.exists():
            skipped += 1
            continue

        duration = rec.get("duration", 0)
        if duration > max_duration:
            skipped += 1
            continue

        # Known corrupted file
        if "U_b8a4e8220e65219b" in str(audio_path):
            skipped += 1
            continue

        text = normalize_ipa(rec["phonetic_text"])
        if not text:
            skipped += 1
            continue

        # Replace spaces with word boundary token
        text = text.replace(" ", "|")

        dataset.append({
            "audio_path": str(audio_path),
            "text": text,
            "duration": duration,
            "utterance_id": rec.get("utterance_id", f"U_{i}"),
        })

    print(f"Prepared {len(dataset)} samples, skipped {skipped}")
    return dataset


@dataclass
class DataCollatorCTC:
    """Data collator for CTC training with dynamic padding."""

    processor: Wav2Vec2Processor
    sample_rate: int = SAMPLE_RATE

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        # Load and process audio
        input_values_list = []
        labels_list = []

        for feat in features:
            speech, sr = sf.read(feat["audio_path"])
            if sr != self.sample_rate:
                speech = librosa.resample(speech, orig_sr=sr, target_sr=self.sample_rate)

            inputs = self.processor(
                speech, sampling_rate=self.sample_rate, return_tensors="pt"
            )
            input_values_list.append(inputs.input_values.squeeze(0))

            with self.processor.as_target_processor():
                labels = self.processor(text=feat["text"], return_tensors="pt")
            labels_list.append(labels.input_ids.squeeze(0))

        # Pad input values
        max_input_len = max(iv.shape[0] for iv in input_values_list)
        padded_inputs = torch.zeros(len(input_values_list), max_input_len)
        attention_mask = torch.zeros(len(input_values_list), max_input_len, dtype=torch.long)
        for i, iv in enumerate(input_values_list):
            padded_inputs[i, : iv.shape[0]] = iv
            attention_mask[i, : iv.shape[0]] = 1

        # Pad labels with -100 (ignore index for CTC loss)
        max_label_len = max(l.shape[0] for l in labels_list)
        padded_labels = torch.full((len(labels_list), max_label_len), -100, dtype=torch.long)
        for i, l in enumerate(labels_list):
            padded_labels[i, : l.shape[0]] = l

        return {
            "input_values": padded_inputs,
            "attention_mask": attention_mask,
            "labels": padded_labels,
        }


def compute_cer(pred, processor: Wav2Vec2Processor) -> dict:
    """Compute Character Error Rate for evaluation."""
    from jiwer import cer

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Decode predictions
    pred_str = processor.batch_decode(pred_ids)

    # Decode labels (replace -100 with pad token)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    # Filter out empty references
    pairs = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
    if not pairs:
        return {"cer": 1.0}

    pred_str, label_str = zip(*pairs)
    score = cer(list(label_str), list(pred_str))
    return {"cer": score}


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to phonetic track data")
    parser.add_argument("--output_dir", type=str, default="model_phonetic", help="Model output directory")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
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

    # Unzip audio
    print("=== Extracting audio ===")
    unzip_audio(data_dir)

    # Load transcripts
    print("=== Loading transcripts ===")
    transcript_file = data_dir / "train_phon_transcripts.jsonl"
    transcripts = load_transcripts(transcript_file)
    print(f"Total transcripts: {len(transcripts)}")

    # Build vocabulary
    print("=== Building vocabulary ===")
    vocab = build_vocab(transcripts)
    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, ensure_ascii=False)

    # Create tokenizer and processor
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLE_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
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

    train_dataset = AudioDataset(train_data)
    val_dataset = AudioDataset(val_data)

    # Load model
    print(f"=== Loading model: {args.model_name} ===")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    # Freeze feature extractor
    model.freeze_feature_encoder()

    # W&B
    run = wandb.init(
        project=args.wandb_project,
        name=args.memo,
        config=vars(args),
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        logging_steps=100,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        dataloader_num_workers=2,
        report_to="wandb",
        max_grad_norm=1.0,
        weight_decay=0.01,
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorCTC(processor=processor)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_cer(pred, processor),
        processing_class=processor.feature_extractor,
    )

    # Train
    print("=== Training ===")
    trainer.train()

    # Evaluate
    print("=== Final evaluation ===")
    metrics = trainer.evaluate()
    print(f"Final CER: {metrics.get('eval_cer', 'N/A')}")
    wandb.log({"final_cer": metrics.get("eval_cer")})

    # Save best model
    print("=== Saving model ===")
    model_save_dir = output_dir / "final_model"
    trainer.save_model(str(model_save_dir))
    processor.save_pretrained(str(model_save_dir))

    # Save vocab separately for submission
    with open(model_save_dir / "vocab.json", "w") as f:
        json.dump(vocab, f, ensure_ascii=False)

    print(f"Model saved to: {model_save_dir}")
    print(f"Model size: {sum(f.stat().st_size for f in model_save_dir.rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")

    run.finish()


if __name__ == "__main__":
    main()
