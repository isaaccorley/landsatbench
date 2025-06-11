from collections.abc import Callable, Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from matplotlib.figure import Figure
from torch import Tensor
from torchgeo.datasets import EuroSAT, RGBBandsMissingError
from torchgeo.datasets.utils import percentile_normalization

dataset_statistics = {
    "min": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "max": torch.tensor([62584.0, 63593.0, 65454.0, 65454.0, 64960.0, 65454.0, 65454.0]),
    "mean": torch.tensor(
        [
            8515.3212890625,
            8829.6005859375,
            9795.9208984375,
            9891.6552734375,
            15499.8837890625,
            13146.115234375,
            11004.2119140625,
        ]
    ),
    "std": torch.tensor(
        [
            3938.755859375,
            4057.7177734375,
            4205.96044921875,
            4434.4228515625,
            6150.6396484375,
            4923.02001953125,
            4044.071044921875,
        ]
    ),
}


def eurosatL_transform(sample: dict[str, Tensor]) -> dict[str, Tensor]:
    """Resize image to 22x22 and remove the last band."""
    sample["image"] = sample["image"].clip(0, None)
    sample["image"] = TF.resize(sample["image"], [22, 22])
    sample["image"] = sample["image"][
        :-1, ...
    ]  # Exclude ST_B10 since this was not used in SSL4EO-L
    return sample


class EuroSATL(EuroSAT):
    url = "https://hf.co/datasets/isaaccorley/eurosat-l/resolve/40282cda59fa453cf4a7a3ccaf4dfc6b353eb58b/"
    filename = "eurosat-l.zip"
    md5 = "559090a4cd3a73a9e3964e2329ab029c"
    base_dir = "eurosat-l"
    all_band_names = ("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10")
    rgb_bands = ("SR_B4", "SR_B3", "SR_B2")
    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: Sequence[str] = BAND_SETS["all"],
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = eurosatL_transform,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        if transforms is None:
            transforms = eurosatL_transform
        super().__init__(root, split, bands, transforms, download, checksum)

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

        label = cast(int, sample["label"].item())
        label_class = self.classes[label]

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = cast(int, sample["prediction"].item())
            prediction_class = self.classes[prediction]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")
        if show_titles:
            title = f"Label: {label_class}"
            if showing_predictions:
                title += f"\nPrediction: {prediction_class}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
