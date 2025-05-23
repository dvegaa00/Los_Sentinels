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