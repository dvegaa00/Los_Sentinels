import os
import torch
import rasterio
import numpy as np
from torch.utils.data import DataLoader
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
import pytorch_lightning as pl

from datasets.forest_segmentation import SentinelBinarySegmentationDataset

# --- Dataloader Setup ---
def build_dataloaders(root, fold=0):
    part_train = "part_1" if fold == 0 else "part_2"
    part_val = "part_2" if fold == 0 else "part_1"

    train_dataset = SentinelBinarySegmentationDataset(os.path.join(root, part_train))
    val_dataset = SentinelBinarySegmentationDataset(os.path.join(root, part_val))

    train_sampler = RandomGeoSampler(train_dataset, size=256, length=1000)
    val_sampler = GridGeoSampler(val_dataset, size=256, stride=256)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=8, num_workers=4)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=8, num_workers=4)

    return train_loader, val_loader