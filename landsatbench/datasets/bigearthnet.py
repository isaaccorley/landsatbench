import json
import os
from collections.abc import Callable
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torchvision.transforms.functional as TF
from matplotlib.figure import Figure
from torch import Tensor
from torchgeo.datasets import BigEarthNet
from torchgeo.datasets.utils import extract_archive, percentile_normalization

dataset_statistics = {
    "min": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "max": torch.tensor(
        [63979.91796875, 65330.30859375, 65454.0, 65454.0, 65454.0, 65454.00390625, 65454.00390625]
    ),
    "mean": torch.tensor(
        [
            9405.1943359375,
            9649.6767578125,
            10425.685546875,
            10444.5888671875,
            16067.626953125,
            12699.18359375,
            10596.474609375,
        ]
    ),
    "std": torch.tensor(
        [
            6016.34375,
            6095.47216796875,
            5839.4619140625,
            6068.49267578125,
            6271.4404296875,
            4256.0458984375,
            3130.7763671875,
        ]
    ),
}


def bigearthnetL_transform(sample: dict[str, Tensor]) -> dict[str, Tensor]:
    """Resize image to 40x40 and remove the last band."""
    sample["image"] = sample["image"].clip(0, None)
    sample["image"] = TF.resize(sample["image"], [40, 40])
    return sample


class BigEarthNetL(BigEarthNet):
    url = "https://hf.co/datasets/isaaccorley/bigearthnet-l/resolve/main/{}"
    filename = "bigearthnet-l.tar.gz"
    md5 = "559090a4cd3a73a9e3964e2329ab029c"
    base_dir = "bigearthnet-l"
    all_band_names = ("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7")
    rgb_bands = ("SR_B4", "SR_B3", "SR_B2")
    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}
    splits_metadata = {
        "train": {
            "url": "https://huggingface.co/datasets/isaaccorley/bigearthnet-l/resolve/main/bigearthnet-train.csv",
            "filename": "bigearthnet-train.csv",
        },
        "val": {
            "url": "https://huggingface.co/datasets/isaaccorley/bigearthnet-l/resolve/main/bigearthnet-val.csv",
            "filename": "bigearthnet-val.csv",
        },
        "test": {
            "url": "https://huggingface.co/datasets/isaaccorley/bigearthnet-l/resolve/main/bigearthnet-test.csv",
            "filename": "bigearthnet-test.csv",
        },
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        num_classes: int = 19,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]]
        | None = bigearthnetL_transform,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        if transforms is None:
            transforms = bigearthnetL_transform
        super().__init__(root, split, "s2", num_classes, transforms, download, checksum)
        split_filepath = os.path.join(self.root, self.splits_metadata[split]["filename"])
        with open(split_filepath) as f:
            paths = [line.split(",")[0] for line in f.read().strip().splitlines()]

        self.images = [
            os.path.join(self.root, self.base_dir, "images", f"{path}.tif") for path in paths
        ]
        self.labels = [
            os.path.join(self.root, self.base_dir, "labels", f"{path}_labels_metadata.json")
            for path in paths
        ]

    def _verify(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        archive_path = os.path.join(self.root, self.filename)
        if not os.path.exists(archive_path) and self.download:
            print(f"Downloading {archive_path}")
            urlretrieve(self.url.format(self.filename), archive_path)
            extract_archive(archive_path)
        else:
            print(f"Archive {archive_path} already exists. Skipping download.")
        for split in self.splits_metadata:
            split_filepath = os.path.join(self.root, self.splits_metadata[split]["filename"])
            if not os.path.exists(split_filepath):
                urlretrieve(self.splits_metadata[split]["url"], split_filepath)

    def _load_folders(self):
        pass

    def __len__(self) -> int:
        return len(self.images)

    def _load_image(self, index: int) -> Tensor:
        path = self.images[index]
        with rasterio.open(path) as dataset:
            array = dataset.read().astype(np.int32)
        tensor = torch.from_numpy(array).float()
        return tensor

    def _load_target(self, index: int) -> Tensor:
        path = self.labels[index]
        with open(path) as f:
            labels = json.load(f)["labels"]

        # labels -> indices
        indices = [self.class2idx[label] for label in labels]

        # Map 43 to 19 class labels
        if self.num_classes == 19:
            indices_optional = [self.label_converter.get(idx) for idx in indices]
            indices = [idx for idx in indices_optional if idx is not None]

        target = torch.zeros(self.num_classes, dtype=torch.long)
        target[indices] = 1
        return target

    def plot(
        self, sample: dict[str, Tensor], show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        image = np.rollaxis(sample["image"][[3, 2, 1]].numpy(), 0, 3)
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
