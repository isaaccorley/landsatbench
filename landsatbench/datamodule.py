from collections.abc import Callable

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .datasets import LC100L, BigEarthNetL, EuroSATL


class LandsatDataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: str,
        root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        transforms: Callable | None = None,
        dataset_transforms: Callable | None = None,
        download: bool = False,
    ):
        super().__init__()
        assert name in ["eurosat", "bigearthnet", "lc100"], (
            "Dataset name must be in (eurosat, bigearthnet, lc100)"
        )
        self.name = name
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.dataset_transforms = dataset_transforms
        self.download = download

    def prepare_data(self):
        if self.name == "eurosat":
            self.dataset_class = EuroSATL
            self.num_classes = 10
            self.num_bands = 7
            # Total 8 bands in Landsat, but ST_B10 is not used in SSL4EO-L. Original EuroSAT has 13 bands.
        elif self.name == "bigearthnet":
            self.dataset_class = BigEarthNetL
            self.num_classes = 19
            self.num_bands = 7
        elif self.name == "lc100":
            self.dataset_class = LC100L
            self.num_classes = 23
            self.num_bands = 7
        else:
            raise ValueError(f"Dataset {self.name} not supported")

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self.dataset_class(
                root=self.root,
                split="train",
                transforms=self.dataset_transforms,
                download=self.download,
            )
            self.val_dataset = self.dataset_class(
                root=self.root,
                split="val",
                transforms=self.dataset_transforms,
                download=self.download,
            )

        elif stage == "test":
            self.test_dataset = self.dataset_class(
                root=self.root,
                split="test",
                transforms=self.dataset_transforms,
                download=self.download,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if self.transforms is not None:
            batch["image"] = self.transforms(batch["image"])
        return batch
