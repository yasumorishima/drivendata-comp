"""What's Up, Docs? — LoRA fine-tune of a small instruct LLM to write the authors' abstract
(runs as a Kaggle script kernel on one T4).

Input (Kaggle dataset from `extractive.py material`): material.csv with paper_id, split
(train = 900 papers with out-of-fold material / val = 100 held-out papers / test), extract,
material (top sentences by the extractive model), opening (first 350 words), reference.
Writes to /kaggle/working: val_scores.txt, preds.csv, submission.csv (best variant on val).
"""

import glob
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter

# Kaggle's image ships an old torchao that the latest peft refuses to import
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=False)

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")  # Apache-2.0
OUT = os.environ.get("LLM_OUT", "/kaggle/working")
INPUT = os.environ.get("LLM_INPUT", "/kaggle/input/**/material.csv")
LIMIT = int(os.environ.get("LLM_LIMIT", "0"))  # quick test
EPOCHS = 2
ACCUM = 8
LR = 2e-4
MAX_IN = 1400  # prompt tokens
MAX_OUT = 360  # abstract tokens (train target and generation)
BATCH = 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SYSTEM = "You write the abstract of a social science paper the way its authors would."


def tokens(x):
    return re.findall(r"[a-z0-9]+", str(x).lower())


def rouge2(ref, pred):
    t, p = tokens(ref), tokens(pred)
    r, q = Counter(zip(t, t[1:])), Counter(zip(p, p[1:]))
    m = sum((r & q).values())
    if m == 0:
        return 0.0
    P, R = m / sum(q.values()), m / sum(r.values())
    return 2 * P * R / (P + R)


def prompt(tok, opening, material):
    words = material.split()
    while True:
        user = (f"OPENING OF THE PAPER:\n{opening}\n\nKEY SENTENCES FROM THE PAPER:\n{' '.join(words)}"
                "\n\nWrite the abstract.")
        text = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) <= MAX_IN or len(words) < 50:
            return ids[-MAX_IN:]
        words = words[: int(len(words) * 0.85)]


def main():
    torch.manual_seed(0)
    random.seed(0)
    df = pd.read_csv(glob.glob(INPUT, recursive=True)[0]).fillna("")
    if LIMIT:
        df = pd.concat([df[df.split == s].head(LIMIT) for s in ("train", "val", "test")], ignore_index=True)
    print("rows", df.split.value_counts().to_dict(), "GPUs", torch.cuda.device_count(), flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # fp32 master weights, fp16 compute (T4 has no bf16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEV)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    prompts = [prompt(tok, o, m) for o, m in zip(df.opening, df.material)]
    tr = [i for i in range(len(df)) if df.split[i] == "train"]
    examples = []
    for i in tr:
        y = tok(df.reference[i] + tok.eos_token).input_ids[:MAX_OUT]
        examples.append((prompts[i] + y, [-100] * len(prompts[i]) + y))
    print("train examples", len(examples), "mean tokens", sum(len(e[0]) for e in examples) / max(len(examples), 1), flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    total = EPOCHS * math.ceil(len(examples) / ACCUM)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    model.train()
    step, t0, run = 0, time.time(), []
    for ep in range(EPOCHS):
        random.shuffle(examples)
        for k, (ids, lab) in enumerate(examples):
            x = torch.tensor([ids]).to(DEV)
            y = torch.tensor([lab]).to(DEV)
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                loss = model(input_ids=x, labels=y).loss / ACCUM
            scaler.scale(loss).backward()
            run.append(loss.item() * ACCUM)
            if (k + 1) % ACCUM == 0 or k + 1 == len(examples):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 10 == 0:
                    print(f"ep {ep} step {step}/{total} loss {sum(run[-80:]) / len(run[-80:]):.3f} {time.time() - t0:.0f}s", flush=True)

    model.eval()
    model.config.use_cache = True
    gen_idx = [i for i in range(len(df)) if df.split[i] != "train"]
    gen_idx.sort(key=lambda i: (df.split[i] != "val", len(prompts[i])))
    out = {}
    t0 = time.time()
    for k in range(0, len(gen_idx), BATCH):
        idx = gen_idx[k:k + BATCH]
        L = max(len(prompts[i]) for i in idx)
        x = torch.tensor([[tok.pad_token_id] * (L - len(prompts[i])) + prompts[i] for i in idx]).to(DEV)
        att = torch.tensor([[0] * (L - len(prompts[i])) + [1] * len(prompts[i]) for i in idx]).to(DEV)
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            g = model.generate(input_ids=x, attention_mask=att, max_new_tokens=MAX_OUT, do_sample=False,
                               no_repeat_ngram_size=6, pad_token_id=tok.pad_token_id)
        for i, row in zip(idx, g):
            out[i] = tok.decode(row[L:], skip_special_tokens=True).replace("\n", " ").strip()
        print(f"gen {k + len(idx)}/{len(gen_idx)} {time.time() - t0:.0f}s", flush=True)
        if k == 0:
            print("SAMPLE:", out[idx[0]][:600], flush=True)

    g = df.loc[gen_idx].copy()
    g["lora"] = [out[i] for i in gen_idx]
    g["lora_ext"] = [(a + " " + " ".join(b.split()[:60])).strip() for a, b in zip(g.lora, g.extract)]
    variants = ["lora", "extract", "lora_ext"]
    val = g[g.split == "val"]
    lines = []
    for v in variants:
        sc = [rouge2(r, p) for r, p in zip(val.reference, val[v])]
        lines.append((sum(sc) / max(len(sc), 1), v, val[v].str.split().str.len().median()))
    # how often the LLM beats the extract, paper by paper
    wins = sum(rouge2(r, a) > rouge2(r, b) for r, a, b in zip(val.reference, val.lora, val.extract))
    lines.sort(reverse=True)
    report = "\n".join(f"{v}: ROUGE-2 F1 {sc:.4f} (median {w:.0f} words)" for sc, v, w in lines)
    report += f"\nlora beats extract on {wins}/{len(val)} val papers\nmodel {MODEL} epochs {EPOCHS} lr {LR}"
    print(report, flush=True)
    open(f"{OUT}/val_scores.txt", "w").write(report + "\n")
    g.drop(columns=["material", "opening"]).to_csv(f"{OUT}/preds.csv", index=False)
    best = lines[0][1]
    test = g[g.split == "test"]
    pd.DataFrame({"paper_id": test.paper_id, "summary": test[best]}).to_csv(f"{OUT}/submission.csv", index=False)
    print("submission variant:", best, flush=True)


if __name__ == "__main__":
    main()
