import json
import os
import sys

sys.path.append("..")

import torch
from landsatbench.datasets import LC100L
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.join("..", "data")
NUM_WORKERS = 8
BATCH_SIZE = 32

if __name__ == "__main__":
    dataset = LC100L(root=ROOT, split="train")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    stats = {}
    x = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        x.append(batch["image"])

    x = torch.cat(x, dim=0)
    stats["min"] = x.amin(dim=(0, 2, 3)).numpy().tolist()
    stats["max"] = x.amax(dim=(0, 2, 3)).numpy().tolist()
    stats["mean"] = x.mean(dim=(0, 2, 3)).numpy().tolist()
    stats["std"] = x.std(dim=(0, 2, 3)).numpy().tolist()

    with open("lc100_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
