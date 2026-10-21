"""What's Up, Docs? — LLM abstract writer (runs as a Kaggle script kernel on GPU).

Input (Kaggle dataset, built by .github/workflows/kaggle-run.yml from extractive.py material):
  material.csv: paper_id, split (val/test), extract (v2 extractive summary), material (top
  sentences), opening (first 350 words), reference (val only).
Writes to /kaggle/working:
  preds.csv (every variant for every paper), val_scores.txt, submission.csv (best variant on val).
"""

import glob
import os
import re
import time
from collections import Counter

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")  # Apache-2.0
OUT = os.environ.get("LLM_OUT", "/kaggle/working")
INPUT = os.environ.get("LLM_INPUT", "/kaggle/input/**/material.csv")
LIMIT = int(os.environ.get("LLM_LIMIT", "0"))  # for a quick local test
BATCH = 12
MAX_NEW = 330

PROMPT = """Below are the opening of a social science research paper and the sentences that best describe its contribution.

Write the paper's abstract: one paragraph of about 170 words, in plain text, as the authors would write it \
(for example "This article examines ...", "Using data from ..., we find ..."). State the question, the data or \
method, the main findings and the implications. Reuse the paper's own key terms and phrases wherever possible. \
Do not add a title, headings, citations or anything that is not in the paper.

OPENING OF THE PAPER:
{opening}

KEY SENTENCES:
{material}

ABSTRACT:"""


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


def main():
    path = glob.glob(INPUT, recursive=True)[0]
    df = pd.read_csv(path).fillna("")
    df = df[df.split != "train"].reset_index(drop=True)
    if LIMIT:
        df = pd.concat([df[df.split == "val"].head(LIMIT), df[df.split == "test"].head(LIMIT)], ignore_index=True)
    print("input", path, df.shape, "GPUs", torch.cuda.device_count(), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                 device_map="auto" if torch.cuda.is_available() else None)
    model.eval()

    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": "You are an expert social scientist who writes precise academic abstracts."},
         {"role": "user", "content": PROMPT.format(opening=o, material=m)}],
        tokenize=False, add_generation_prompt=True) for o, m in zip(df.opening, df.material)]

    # val first so its score shows up early in the log
    order = sorted(range(len(df)), key=lambda i: (df.split[i] != "val", len(prompts[i])))
    out = [""] * len(df)
    t0 = time.time()
    for k in range(0, len(order), BATCH):
        idx = order[k:k + BATCH]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, repetition_penalty=1.05,
                                 pad_token_id=tok.eos_token_id)
        for i, g in zip(idx, gen):
            out[i] = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"{k + len(idx)}/{len(df)} done, {time.time() - t0:.0f}s", flush=True)
        if k == 0:
            print("SAMPLE:", out[idx[0]][:600], flush=True)

    df["llm"] = [re.sub(r"^\s*(abstract|summary)\s*[:.]\s*", "", o, flags=re.I).replace("\n", " ").strip() for o in out]
    # variants: LLM alone; extractive alone; LLM followed by extractive sentences (more shared word pairs)
    df["llm_ext"] = [(a + " " + " ".join(b.split()[:80])).strip() for a, b in zip(df.llm, df.extract)]
    df["ext_llm"] = [(b + " " + a).strip() for a, b in zip(df.llm, df.extract)]
    variants = ["llm", "extract", "llm_ext", "ext_llm"]
    val = df[df.split == "val"]
    lines = []
    for v in variants:
        sc = sum(rouge2(r, p) for r, p in zip(val.reference, val[v])) / max(len(val), 1)
        words = val[v].str.split().str.len().median()
        lines.append((sc, v, words))
    lines.sort(reverse=True)
    report = "\n".join(f"{v}: ROUGE-2 F1 {sc:.4f} (median {w:.0f} words)" for sc, v, w in lines)
    print(report, flush=True)
    open(f"{OUT}/val_scores.txt", "w").write(report + "\n")
    df.drop(columns=["material", "opening"]).to_csv(f"{OUT}/preds.csv", index=False)
    best = lines[0][1]
    test = df[df.split == "test"]
    pd.DataFrame({"paper_id": test.paper_id, "summary": test[best]}).to_csv(f"{OUT}/submission.csv", index=False)
    print("submission variant:", best, flush=True)


if __name__ == "__main__":
    main()
