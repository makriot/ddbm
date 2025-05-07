import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from torchvision import transforms as tr
from PIL import Image
import random
from torch.utils.data import DataLoader
from datasets import InfiniteBatchSampler


def get_channel_statistics(dataset):
    """
    Function for obtaining channel statistics (mean and deviation) for a dataset
    """
    channel_sum = np.zeros(3)
    channel_sq_sum = np.zeros(3)

    num_pixels = 0

    for img in tqdm(dataset):
        img = np.array(img) / 255.0
        channel_sum += img.mean(axis=(0, 1))
        channel_sq_sum += (img**2).mean(axis=(0, 1))
        num_pixels += 1

    channel_mean = channel_sum / num_pixels
    channel_std = np.sqrt(channel_sq_sum/num_pixels - channel_mean**2)

    return channel_mean, channel_std


# def get_transforms(mean, std, img_size=256):
def get_transforms(img_size=256):
    train_transform = tr.Compose([
        tr.ToPILImage(),
        tr.Resize(img_size),
        tr.ToTensor(),
        # tr.Lambda(lambda x: x / 255.0),
        # tr.Normalize(mean, std)
    ])

    test_transform = tr.Compose([
        tr.ToPILImage(),
        tr.Resize(img_size),
        tr.ToTensor(),
        # tr.Lambda(lambda x: x / 255.0),
        # tr.Normalize(mean, std)
    ])
    
    def de_normalize(tensor, normalized=True):
        if len(tensor.shape) == 3:
            tmp = tensor.cpu() * torch.from_numpy(std).reshape(3,1,1) + torch.from_numpy(mean).reshape(3,1,1)
            return tmp.permute(1, 2, 0)
        elif len(tensor.shape) == 4:
            tmp = tensor.cpu() * torch.from_numpy(std).reshape(1,3,1,1) + torch.from_numpy(mean).reshape(1,3,1,1)
            return tmp.permute(0, 2, 3, 1)
    
    return train_transform, test_transform, de_normalize


def flip(img):
    if isinstance(img, torch.Tensor):
        return img.flip(-1)
    return img.transpose(Image.FLIP_LEFT_RIGHT)


class ImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode=None):
        """
        mode: one of "a", "b", None
        """
        super().__init__()
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = cv2.imread(img_path)[:, :, ::-1]  # Convert BGR to RGB
        if self.mode == "a":
            image = image[:, :image.shape[1]//2, :]
        elif self.mode == "b":
            image = image[:, image.shape[1]//2:, :]
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)


class PairedDataset(Dataset):
    def __init__(self, root_dir, transform_a=None, transform_b=None, random_flip=False, order=False):
        super().__init__()
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        self.transform_a = transform_a
        self.transform_b = transform_b
        self.random_flip = random_flip
        self.order = order

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = cv2.imread(img_path)[:, :, ::-1]  # Convert BGR to RGB
        image_a = image[:, :image.shape[1]//2, :]
        image_b = image[:, image.shape[1]//2:, :]
        if self.transform_a:
            image_a = self.transform_a(image_a)
        if self.transform_b:
            image_b = self.transform_b(image_b)
        
        if self.random_flip and random.random() < 0.5:
            image_a = flip(image_a)
            image_b = flip(image_b)

        if self.order:
            return image_b, image_a, index
        return image_a, image_b, index  # index is needed for sampling

    def __len__(self):
        return len(self.image_paths)


@dataclass
class ImagesDatasetsClass:
    train_a: PairedDataset
    train_b: PairedDataset
    val_a: PairedDataset
    val_b: PairedDataset

@dataclass
class PairedDatasetsClass:
    train: PairedDataset
    val: PairedDataset


def load_data(
    data_dir,
    dataset,
    batch_size,
    image_size,
    deterministic=False,
    include_test=False,
    seed=42,
    num_workers=2,
    order=False,
):
    target_dir = os.path.join(data_dir, dataset)

    ds = ImagesDatasetsClass(
        train_a = ImagesDataset(os.path.join(target_dir, "train"), mode="a"),
        train_b = ImagesDataset(os.path.join(target_dir, "train"), mode="b"),
        val_a = ImagesDataset(os.path.join(target_dir, "val"), mode="a"),
        val_b = ImagesDataset(os.path.join(target_dir, "val"), mode="b")
    )

    train_transform_a, test_transform_a, de_normalize_a = get_transforms(img_size=image_size)
    train_transform_b, test_transform_b, de_normalize_b = get_transforms(img_size=image_size)

    ds_pairs = PairedDatasetsClass(
        train = PairedDataset(os.path.join(target_dir, "train"),
                              transform_a=train_transform_a,
                              transform_b=train_transform_b,
                              random_flip=True,
                              order=order),
        val = PairedDataset(os.path.join(target_dir, "val"),\
                            transform_a=test_transform_a,
                            transform_b=test_transform_b,
                            random_flip=False,
                            order=order),
    )

    train_loader = DataLoader(
        dataset=ds_pairs.train, num_workers=num_workers, # pin_memory=True,
        batch_sampler=InfiniteBatchSampler(
            dataset_len=len(ds_pairs.train), batch_size=batch_size, seed=seed, shuffle=not deterministic
        )
    )

    val_loader = DataLoader(
        dataset=ds_pairs.val, num_workers=num_workers, # pin_memory=True,
        batch_sampler=InfiniteBatchSampler(
            dataset_len=len(ds_pairs.val), batch_size=batch_size, seed=seed, shuffle=not deterministic
        )
    )
    
    if include_test:
        ds_test = PairedDataset(os.path.join(target_dir, "test"),
                                transform_a=train_transform_a,
                                transform_b=train_transform_b,
                                random_flip=False,
                                order=order)

        test_loader = DataLoader(
            dataset=ds_test, num_workers=num_workers, # pin_memory=True,
            batch_sampler=InfiniteBatchSampler(
                dataset_len=len(ds_test), batch_size=batch_size, seed=seed, shuffle=not deterministic
        )
    )
        return train_loader, val_loader, test_loader, de_normalize_a, de_normalize_b
    else:
        return train_loader, val_loader, de_normalize_a, de_normalize_b
