import sys

sys.path.append("..")

import os

import kornia.augmentation as K
import numpy as np
import timm
import torch
from landsatbench.datamodule import LandsatDataModule
from landsatbench.datasets.lc100 import dataset_statistics
from landsatbench.embed import extract_features
from torchgeo.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    ViTSmall16_Weights,
    resnet18,
    resnet50,
    vit_small_patch16_224,
)
from tqdm import tqdm

# Imagenet transforms
mins = dataset_statistics["min"]
maxs = dataset_statistics["max"]
means = dataset_statistics["mean"] / maxs
stds = dataset_statistics["std"] / maxs
imagenet_transforms = K.ImageSequential(
    K.Normalize(mean=0.0, std=maxs), K.Normalize(mean=means, std=stds), keepdim=True
)

# SSL4EO-L transforms
min = 7272.72727272727272727272
max = 18181.81818181818181818181
ssl4eol_transforms = K.ImageSequential(
    K.Normalize(mean=min, std=1.0), K.Normalize(mean=0.0, std=max - min), keepdim=True
)

models = {
    "resnet18-imagenet": dict(
        model=timm.create_model,
        transforms=imagenet_transforms,
        kwargs=dict(model_name="resnet18", pretrained=True),
    ),
    "resnet50-imagenet": dict(
        model=timm.create_model,
        transforms=imagenet_transforms,
        kwargs=dict(model_name="resnet50", pretrained=True),
    ),
    "vits-imagenet": dict(
        model=timm.create_model,
        transforms=imagenet_transforms,
        kwargs=dict(model_name="vit_small_patch16_224", pretrained=True),
    ),
    "resnet18-ssl4eol-moco": dict(
        model=resnet18,
        transforms=ssl4eol_transforms,
        kwargs=dict(weights=ResNet18_Weights.LANDSAT_OLI_SR_MOCO),
    ),
    "resnet18-ssl4eol-simclr": dict(
        model=resnet18,
        transforms=ssl4eol_transforms,
        kwargs=dict(weights=ResNet18_Weights.LANDSAT_OLI_SR_SIMCLR),
    ),
    "resnet50-ssl4eol-moco": dict(
        model=resnet50,
        transforms=ssl4eol_transforms,
        kwargs=dict(weights=ResNet50_Weights.LANDSAT_OLI_SR_MOCO),
    ),
    "resnet50-ssl4eol-simclr": dict(
        model=resnet50,
        transforms=ssl4eol_transforms,
        kwargs=dict(weights=ResNet50_Weights.LANDSAT_OLI_SR_SIMCLR),
    ),
    "vits-ssl4eol-moco": dict(
        model=vit_small_patch16_224,
        transforms=ssl4eol_transforms,
        kwargs=dict(weights=ViTSmall16_Weights.LANDSAT_OLI_SR_MOCO),
    ),
    "vits-ssl4eol-simclr": dict(
        model=vit_small_patch16_224,
        transforms=ssl4eol_transforms,
        kwargs=dict(weights=ViTSmall16_Weights.LANDSAT_OLI_SR_SIMCLR),
    ),
}


if __name__ == "__main__":
    output_dir = "../embeddings"
    os.makedirs(output_dir, exist_ok=True)

    k = 5
    device = torch.device("mps")

    root = "../data"
    dm = LandsatDataModule(name="lc100", root=root, batch_size=32, num_workers=12, download=False)
    dm.prepare_data()

    for name, v in tqdm(models.items(), total=len(models)):
        print(f"Embedding {name}...")
        model = v["model"](**v["kwargs"], num_classes=0, in_chans=7).to(device)
        transforms = v["transforms"]

        dm.setup("fit")
        x_train, y_train = extract_features(model, dm.train_dataloader(), device, transforms)

        dm.setup("test")
        x_test, y_test = extract_features(model, dm.test_dataloader(), device, transforms)

        filename = os.path.join(output_dir, f"lc100-{name}.npz")
        np.savez(
            filename,
            x_train=x_train,
            y_train=y_train.astype(np.int16),
            x_test=x_test,
            y_test=y_test.astype(np.int16),
        )
