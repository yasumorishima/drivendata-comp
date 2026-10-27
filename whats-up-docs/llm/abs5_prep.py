"""Input for cpu_abstract5.py (fifth abstract, llm-abs5-v1): the opening, the start of the data / method section
and the start of the results section, where abstracts take their "Using data from ..., we find ..." part.

  python whats-up-docs/llm/abs5_prep.py <data_dir> <out.csv>
Columns: paper_id, split, opening, material.
"""

import re
import sys

import pandas as pd

OPEN, PART = 150, 325
METHOD = re.compile(r"data|method|research design|empirical|sample|measure|case|approach|model", re.I)
RESULT = re.compile(r"result|finding|analysis|evidence|estimat", re.I)


def sections(text):
    text = str(text)
    heads = list(re.finditer(r"^#+ .*$", text, re.M))
    spans = [(h.group(), text[h.end(): heads[k + 1].start() if k + 1 < len(heads) else len(text)]) for k, h in enumerate(heads)]
    pick = []
    for rx in (METHOD, RESULT):
        body = next((b for h, b in spans if rx.search(h) and len(b.split()) > 50), "")
        pick.append(" ".join(body.split()[:PART]))
    if not any(pick):  # no usable headings: the middle of the paper
        w = text.split()
        mid = len(w) // 2
        pick = [" ".join(w[mid - PART: mid + PART])]
    return " ... ".join(p for p in pick if p)


if __name__ == "__main__":
    d, out = sys.argv[1].rstrip("/") + "/", sys.argv[2]
    rows = []
    for split, f in [("test", "test_features.csv"), ("train", "train.csv")]:
        df = pd.read_csv(d + f)
        for p, t in zip(df.paper_id, df.text):
            rows.append({"paper_id": p, "split": split, "opening": " ".join(str(t).split()[:OPEN]), "material": sections(t)})
    pd.DataFrame(rows).to_csv(out, index=False)
    print("rows", len(rows))
