# datasets/forest_segmentation.py
import os
import torch
from torchgeo.datasets import NonGeoDataset, GeoDataset, RasterDataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import rasterio
import numpy as np

class Sentinel2Images(RasterDataset):
    filename_glob = '*_image.tif'
    is_image = True
    separate_files = False
    all_bands = ('B02', 'B03', 'B04', 'B08')
    rgb_bands = ('B04', 'B03', 'B02')

class Sentinel2Masks(RasterDataset):
    filename_glob = '*_mask.tif'
    is_image = False
    separate_files = False

class ForestSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, zones, patch_size=256, transform=None, split="train"):
        """
        root_dir: path to 'processed/'
        zones: list of zona_x folders to include (e.g., ['zona_1', 'zona_2'])
        part: 'part_1' or 'part_2' (used for cross-val split)
        """
        self.samples = []
        self.patch_size = patch_size
        self.transform = transform

        for zone_dir, _ in zones:
            for fname in os.listdir(zone_dir):
                if fname.endswith("_image.tif"):
                    mask_name = fname.replace("_image.tif", "_mask.tif")
                    img_path = os.path.join(zone_dir, fname)
                    mask_path = os.path.join(zone_dir, mask_name)
                    self.samples.append((img_path, mask_path))

        self.split = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Read image and mask
        with rasterio.open(img_path) as img_file:
            img = img_file.read().astype(np.float32)  # (C, H, W)
            img[np.isnan(img)] = 0  # Replace NaNs
        with rasterio.open(mask_path) as mask_file:
            mask = mask_file.read(1)  # single-channel mask (H, W)

        # Normalize image
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask).long()

        # Random crop
        i = torch.randint(0, img.shape[1] - self.patch_size + 1, (1,)).item()
        j = torch.randint(0, img.shape[2] - self.patch_size + 1, (1,)).item()
        img = img[:, i:i+self.patch_size, j:j+self.patch_size]
        mask = mask[i:i+self.patch_size, j:j+self.patch_size]

        # Convert mask to binary (1 = forest, 0 = not forest)
        mask = torch.where(mask == 1, 1, torch.where(mask == 2, 0, 255))  # 255 = ignore index

        if self.transform:
            img = self.transform(img)

        sample = {'image': img,
                  'mask': mask}

        return sample


class SentinelBinarySegmentationDataset(GeoDataset):
    def __init__(self, root, transforms=None):
        super().__init__()
        self.root = root
        self.transforms = transforms

        self.image_paths = []
        self.mask_paths = []

        for part in sorted(os.listdir(root)):
            part_dir = os.path.join(root, part)
            if os.path.isdir(part_dir):
                for zone in sorted(os.listdir(part_dir)):
                    zone_dir = os.path.join(part_dir, zone)
                    image_file = [f for f in os.listdir(zone_dir) if f.endswith('_image.tif')][0]
                    mask_file = [f for f in os.listdir(zone_dir) if f.endswith('_mask.tif')][0]

                    self.image_paths.append(os.path.join(zone_dir, image_file))
                    self.mask_paths.append(os.path.join(zone_dir, mask_file))

        # assume same CRS and transform across all tiles
        with rasterio.open(self.image_paths[0]) as src:
            self.crs = src.crs
            self.transform = src.transform
            self.bounds = src.bounds

    def __getitem__(self, query):
        # TorchGeo expects __getitem__ to accept bounding box queries
        if isinstance(query, dict):  # TorchGeo sampling
            with rasterio.open(self.image_paths[0]) as img_src, rasterio.open(self.mask_paths[0]) as mask_src:
                window = rasterio.windows.from_bounds(**query, transform=self.transform)
                image = img_src.read(window=window)  # shape: (C, H, W)
                mask = mask_src.read(1, window=window)  # shape: (H, W)

        else:
            raise ValueError("Query type not supported")

        # Replace NaNs with 0
        image = np.nan_to_num(image)

        # Map mask from {0,1,2} -> {-1,1} with ignore index (-1)
        mask = np.where(mask == 0, -1, mask - 1)

        sample = {
            'image': torch.from_numpy(image).float(),
            'mask': torch.from_numpy(mask).long(),
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return 10000  # needed by RandomGeoSampler (virtual size)