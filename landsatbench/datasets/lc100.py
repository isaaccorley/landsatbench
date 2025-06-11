import glob
import os
from collections.abc import Callable, Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torchvision.transforms.functional as TF
from matplotlib.figure import Figure
from torch import Tensor
from torchgeo.datasets import CopernicusBenchLC100ClsS3, RGBBandsMissingError, stack_samples
from torchgeo.datasets.utils import percentile_normalization

dataset_statistics = {
    "min": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "max": torch.tensor([65454.0, 65454.0, 65454.0, 65454.0, 65454.0, 65175.0, 61457.0]),
    "mean": torch.tensor(
        [
            10349.7880859375,
            10614.703125,
            11293.564453125,
            11854.380859375,
            14793.0302734375,
            12222.5654296875,
            10831.328125,
        ]
    ),
    "std": torch.tensor(
        [
            10919.978515625,
            10937.5087890625,
            10435.6298828125,
            10778.1865234375,
            10573.8857421875,
            8040.63525390625,
            7096.6103515625,
        ]
    ),
}


def lc100L_transform(sample: dict[str, Tensor]) -> dict[str, Tensor]:
    """Resize image to 224x224."""
    sample["image"] = sample["image"].clip(0, None)
    sample["image"] = TF.resize(sample["image"], [224, 224])
    return sample


class LC100L(CopernicusBenchLC100ClsS3):
    directory = "lc100-l"
    all_band_names = ("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7")
    all_bands = ("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7")
    rgb_bands = ("SR_B4", "SR_B3", "SR_B2")
    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    def __init__(
        self,
        root: str = "data",
        split: Literal["train", "val", "test"] = "train",
        mode: Literal["static", "time-series"] = "static",
        bands: Sequence[str] = BAND_SETS["all"],
        download: bool = False,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = lc100L_transform,
    ) -> None:
        if transforms is None:
            transforms = lc100L_transform
        super().__init__(root, split, mode, bands, transforms, False, False)

        # Filter out image where Landsat was unavailable
        pid = "0669864_-114.00_26.25"
        if pid in self.files.PID.values:
            self.files.drop(self.files[self.files["PID"] == pid].index, inplace=True)
            self.static_files.drop(
                self.static_files[
                    self.static_files[self.static_files.columns[0]].values == pid
                ].index,
                inplace=True,
            )

    def _verify(self):
        pass

    def _load_image(self, path: str) -> dict[str, Tensor]:
        sample: dict[str, Tensor] = {}
        with rasterio.open(path) as f:
            image = f.read(self.band_indices).astype(np.float32)
            sample["image"] = torch.tensor(image)
        return sample

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.files.iloc[index].values
        match self.mode:
            case "static":
                pid, file = self.static_files.iloc[index]
                path = os.path.join(self.root, self.directory, pid, file)
                sample = self._load_image(path)
                sample["label"] = torch.tensor(row[1:].astype(np.int64))
            case "time-series":
                pid = row[0]
                paths = os.path.join(self.root, self.directory, pid, "*.tif")
                samples = [self._load_image(path) for path in sorted(glob.glob(paths))]
                sample = stack_samples(samples)
                sample["label"] = torch.tensor(row[1:].astype(np.int64))

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _onehot_labels_to_names(self, label_mask: np.ndarray) -> list[str]:
        labels = []
        for i, mask in enumerate(label_mask):
            if mask:
                labels.append(self.classes[i])
        return labels

    def plot(
        self, sample: dict[str, Tensor], show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        image = np.take(sample["image"].numpy(), indices=rgb_indices, axis=0)
        image = np.rollaxis(image, 0, 3)
        image = percentile_normalization(image, lower=0, upper=100)
        image = np.clip(image, 0, 1)

        label_mask = sample["label"].numpy().astype(np.bool_)
        labels = self._onehot_labels_to_names(label_mask)

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction_mask = sample["prediction"].numpy().astype(np.bool_)
            predictions = self._onehot_labels_to_names(prediction_mask)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")
        if show_titles:
            title = f"Labels: {', '.join(labels)}"
            if showing_predictions:
                title += f"\nPredictions: {', '.join(predictions)}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
