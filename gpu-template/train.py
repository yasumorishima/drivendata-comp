"""GPU competition training template.

Run in Google Colab:
    !python <competition_dir>/train.py --data_dir data/<track> --output_dir model_<track>
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="model_output")
    parser.add_argument("--wandb_project", type=str, default="drivendata-gpu-comp")
    parser.add_argument("--memo", type=str, default="local")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(project=args.wandb_project, name=args.memo, config=vars(args))

    # --- Load data ---
    # TODO: implement data loading

    # --- Train ---
    # TODO: implement training

    # --- Save model ---
    model_save_dir = output_dir / "final_model"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    # TODO: save model weights

    print(f"Model saved to: {model_save_dir}")
    run.finish()


if __name__ == "__main__":
    main()
