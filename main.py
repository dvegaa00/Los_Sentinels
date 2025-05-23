'''

Example of usage:

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --data_dir /home/srodriguezr2/srodriguezr2_2/retina/Los_Sentinels/data --num_gpus 8 --batch_size 64 --fold 0
CUDA_VISIBLE_DEVICES=3 python main.py --data_dir /home/srodriguezr2/srodriguezr2_2/retina/Los_Sentinels/data --num_gpus 1 --batch_size 8 --fold 0

'''


import torch
import os
from torch.utils.data import DataLoader
from datasets.forest_segmentation import ForestSegmentationDataset, Sentinel2Images, Sentinel2Masks
#from models.unet_model import ForestSegformerModel
from models.unet_model import UnetBinaryModule
from utils.data_split import get_part_zones
from datasets.make_dataloader import build_dataloaders
from utils.collate_fn import collate_fn
from utils.dataset_wrapper import FilteredGeoDatasetWrapper

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
import argparse

from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from torchgeo.trainers import SemanticSegmentationTask

import matplotlib.pyplot as plt
import numpy as np
import logging

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="/home/srodriguezr2/srodriguezr2_2/retina/Los_Sentinels/outputs")
    parser.add_argument("--fold", type=int, choices=[0, 1], required=True)
    return parser.parse_args()

# Visualize and save batch of images and masks
def save_visualizations(batch, prefix="train_batch"):

    # Output directory to save visualizations
    save_dir = "/home/srodriguezr2/srodriguezr2_2/retina/Los_Sentinels/visuals"
    os.makedirs(save_dir, exist_ok=True)

    for i in range(min(len(batch), 4)):  # Save up to 4 samples
        sample = batch[i]
        image = sample["image"]
        mask = sample["mask"]

        # Convert image to numpy
        image_np = image[:3].permute(1, 2, 0).cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()

        # Normalize image for visualization
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

        image_title = "Image (RGB) all NaN" if np.isnan(image_np).all() else "Image (RGB)"
        mask_title = "Mask all backgrounnd" if (mask_np == 0).all() else "Mask"

        # Create subplot
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image_np)
        axs[0].set_title(image_title)
        axs[0].axis("off")

        axs[1].imshow(mask_np, cmap="gray")
        axs[1].set_title(mask_title)
        axs[1].axis("off")

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{prefix}_{i}.png")
        plt.savefig(save_path)
        plt.close(fig)  # Avoid displaying in training loop

def main():
    args = get_args()

    all_zones = get_part_zones(args.data_dir, "part_1") + get_part_zones(args.data_dir, "part_2")
    fold_split = {"train": [], "val": []}

    for zone_path, part in all_zones:
        if (part == "part_1" and args.fold == 0) or (part == "part_2" and args.fold == 1):
            fold_split["val"].append((zone_path, part))
        else:
            fold_split["train"].append((zone_path, part))

    # train_dataset = ForestSegmentationDataset(fold_split["train"])
    # val_dataset = ForestSegmentationDataset(fold_split["val"])
            
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
            
    default_root_dir = os.path.join(args.output_dir)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', dirpath=default_root_dir, save_top_k=1, save_last=True
    )
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=30)
    # logger = TensorBoardLogger(save_dir=default_root_dir, name='first_log')
    logger = CSVLogger(save_dir=default_root_dir, name="logs")
    
    if args.fold == 0:
        train_fold = 1
        test_fold = 2

    else:
        train_fold = 2
        test_fold = 1

    train_root_path_imgs = f"/home/srodriguezr2/srodriguezr2_2/retina/Los_Sentinels/data/sentinel2_binary/processed_organized/fold{train_fold}/images"
    train_root_path_masks = f"/home/srodriguezr2/srodriguezr2_2/retina/Los_Sentinels/data/sentinel2_binary/processed_organized/fold{train_fold}/masks"

    images_train = Sentinel2Images(train_root_path_imgs)
    masks_train = Sentinel2Masks(train_root_path_masks)
    
    train_dataset = images_train & masks_train

    breakpoint()
    # train_dataset = FilteredGeoDatasetWrapper(train_dataset)

    g = torch.Generator().manual_seed(42)
    train_sampler = RandomGeoSampler(train_dataset, size=256, length=30000, generator=g) #Length -> Qué tan grandes serán nuestros steps TODO: Debugear la máscara e imagen para ver si están alineadas
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=collate_fn)

    test_root_path_imgs = f"/home/srodriguezr2/srodriguezr2_2/retina/Los_Sentinels/data/sentinel2_binary/processed_organized/fold{test_fold}/images"
    test_root_path_masks = f"/home/srodriguezr2/srodriguezr2_2/retina/Los_Sentinels/data/sentinel2_binary/processed_organized/fold{test_fold}/masks"

    images_test = Sentinel2Images(test_root_path_imgs)
    masks_test = Sentinel2Masks(test_root_path_masks)

    test_dataset = images_test & masks_test

    test_sampler = GridGeoSampler(test_dataset, size=256, stride=256)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)

    # Fetch a batch and save
    for batch in train_loader:
        batch_unstacked = unbind_samples(batch)
        save_visualizations(batch_unstacked, prefix="train")

        breakpoint()

    task = SemanticSegmentationTask(
                                        model = 'unet',
                                        in_channels=4,
                                        num_classes=5,
                                        loss='ce',
                                        lr=0.0001,
                                        patience=5,
                                    )

    trainer = Trainer(
                        # callbacks=[checkpoint_callback, early_stopping_callback],
                        callbacks=[checkpoint_callback],
                        log_every_n_steps=1,
                        logger=logger,
                        min_epochs=1,
                        max_epochs=args.epochs,
                    )
    
    trainer.fit(task, train_loader, test_loader)


if __name__ == "__main__":
    main()
