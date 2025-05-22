import torch

def collate_fn(batch):
    images = torch.stack([item["image"].clone().detach() for item in batch])  # Convert list of images to tensor
    masks = torch.stack([item["mask"].clone().detach().squeeze(0) for item in batch])  # Convert list of masks to tensor and squeeze extra dimension

    new_batch = {
        "image": images,
        "mask": masks,
    }
    return new_batch