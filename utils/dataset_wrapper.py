from torch.utils.data import Dataset
import torch

class FilteredGeoDatasetWrapper:
    def __init__(self, dataset, max_attempts=10):
        self.dataset = dataset
        self.max_attempts = max_attempts

    def __getitem__(self, index):
        for _ in range(self.max_attempts):
            sample = self.dataset[index]
            # Put your filtering condition here, e.g.
            if self.is_valid_sample(sample):
                return sample
            index = (index + 1) % len(self)
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def is_valid_sample(self, sample):
        # Example validation logic: image not all nan and mask not all zeros
        image = sample['image']
        mask = sample['mask']
        if (image.isnan().all()) or (mask.sum() == 0):
            return False
        return True

    def __getattr__(self, name):
        # Forward any attribute not found here to the wrapped dataset
        return getattr(self.dataset, name)


import fnmatch
import os
import pathlib
import warnings
from glob import iglob
from typing import Iterable

import numpy as np
import rasterio
from torchgeo.datasets.geo import RasterDataset
from torchgeo.datasets.utils import Path, path_is_vsi

from .utils import strip_sas_token


class NodataFilteredRasterDataset(RasterDataset):
    """A RasterDataset that filters out files with only nodata values.

    This class extends the torchgeo RasterDataset by adding functionality to
    filter out files that contain only nodata values. It also handles datasets
    initialized as lists of Virtual File System (VFS) paths containing SAS tokens.

    """

    def _init_(
        self, paths: Path | Iterable[Path] = "data", nodata_value: int = 0, **kwargs
    ):

        self.nodata_value = nodata_value
        self.paths = paths
        self._filtered_files = []  # Initialize filtered file list as empty
        self._all_files = self.files  # Get list of all files
        self._filtered_files = (
            self._filter_files()
        )  # Filter files--now .files will return only files with non-nodata values
        super()._init_(
            paths=paths, **kwargs
        )  # Initialize RasterDataset index using only filtered files

    @property
    def files(self) -> list[Path]:
        """A list of all files in the dataset.

        Returns filtered files if available, otherwise retrieves all files
        matching the dataset filename pattern from the provided paths.

        NOTE: Also handles VFS paths containing SAS tokens by stripping
        the SAS token from the URL before attempting to match the filename.

        Returns:
            All files in the dataset.

        """
        # Return filtered files if available
        if self._filtered_files:
            return self._filtered_files

        # Make iterable
        if isinstance(self.paths, str | pathlib.Path):
            paths: Iterable[Path] = [self.paths]
        else:
            paths = self.paths

        # Using set to remove any duplicates if directories are overlapping
        files: set[Path] = set()
        for path in paths:
            if os.path.isdir(path):
                pathname = os.path.join(path, "**", self.filename_glob)
                files |= set(iglob(pathname, recursive=True))
            elif os.path.isfile(path):
                if fnmatch.fnmatch(str(path), f"*{self.filename_glob}"):
                    files.add(path)
            elif path_is_vsi(path):
                # Strip SAS token from URL for path matching
                stripped_url = strip_sas_token(path)
                if fnmatch.fnmatch(stripped_url, f"*{self.filename_glob}"):
                    files.add(path)
            elif not hasattr(self, "download"):
                warnings.warn(
                    f"Could not find any relevant files for provided path '{path}'. "
                    f"Path was ignored.",
                    UserWarning,
                )

        # Sort the output to enforce deterministic behavior.
        return sorted(files)

    def _filter_files(self):
        """Filter out files containing only nodata values."""
        filtered_files = []

        # Filter files based on nodata_value
        for file in self._all_files:
            with rasterio.open(file) as src:
                data = src.read(1)
                if np.any(data != self.nodata_value):
                    filtered_files.append(file)
                else:
                    print(
                        f"File {file} contains only nodata values and will be ignored."
                    )
        return filtered_files