import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


@torch.inference_mode()
def extract_features(model, dataloader, device, transforms=nn.Identity()):
    x_all, y_all = [], []

    model.eval()
    model = model.to(device)

    for batch in tqdm(dataloader, total=len(dataloader)):
        images = batch["image"].to(device)
        labels = batch["label"].numpy()
        images = transforms(images)

        features = model(images)
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        x_all.append(features)
        y_all.append(labels)

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return x_all, y_all


class ImageStatisticsModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat(
            [
                torch.mean(x, dim=(2, 3)),
                torch.std(x, dim=(2, 3)),
                torch.amax(x, dim=(2, 3)),
                torch.amin(x, dim=(2, 3)),
            ],
            dim=1,
        )
