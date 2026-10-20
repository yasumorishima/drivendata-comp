"""Submit mode for a Kaggle kernel's output: copy its submission.csv after checking the format.

  python whats-up-docs/from_kernel.py submit <kout_dir> <out.csv>
<kout_dir> is ~/drivendata-env/data/kout-<name> (bash tools/cloud_setup.sh kout-<name>).
"""

import sys

import pandas as pd

if __name__ == "__main__":
    _, mode, d, out = sys.argv
    sub = pd.read_csv(d.rstrip("/") + "/submission.csv")
    assert list(sub.columns) == ["paper_id", "summary"] and len(sub) == 345, sub.shape
    sub["summary"] = sub.summary.fillna("").astype(str)
    assert (sub.summary.str.split().str.len() > 20).mean() > 0.95, "too many empty summaries"
    sub.to_csv(out, index=False)
    print("copied", len(sub), "median words", sub.summary.str.split().str.len().median())
