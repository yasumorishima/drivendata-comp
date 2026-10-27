"""Input for cpu_abstract3.py: the opening and the ending (conclusion / discussion section) of every paper.

The first two LLM abstracts (llm-v1, llm-abs2-v1) both read the opening plus the top model sentences, so they
look at the same places. The third reads where the authors restate their findings: the last conclusion-like
section (or, without one, the last words before the references).

  python whats-up-docs/llm/abs3_prep.py <data_dir> <out.csv>
Columns: paper_id, split, opening, material.
"""

import re
import sys

import pandas as pd

OPEN, END = 150, 650
CONCL = re.compile(r"conclu|discussion|summary|implication|final remarks", re.I)
REFS = re.compile(r"^#+\s*(references|bibliography|works cited|notes|acknowledg|appendix)", re.I | re.M)


def ending(text):
    text = str(text)
    m = REFS.search(text)
    body = text[: m.start()] if m and m.start() > len(text) * 0.4 else text
    heads = [h for h in re.finditer(r"^#+ .*$", body, re.M) if CONCL.search(h.group())]
    part = body[heads[-1].start():] if heads else body
    w = part.split()
    return " ".join(w[:END]) if heads else " ".join(w[-END:])


if __name__ == "__main__":
    d, out = sys.argv[1].rstrip("/") + "/", sys.argv[2]
    rows = []
    for split, f in [("test", "test_features.csv"), ("train", "train.csv")]:
        df = pd.read_csv(d + f)
        for p, t in zip(df.paper_id, df.text):
            rows.append({"paper_id": p, "split": split, "opening": " ".join(str(t).split()[:OPEN]), "material": ending(t)})
    pd.DataFrame(rows).to_csv(out, index=False)
    print("rows", len(rows))
