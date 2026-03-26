import os
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from ._base import register_dataset


@register_dataset('bmnist')
class BinaryMNIST(Dataset):
    def __init__(self, root, split, indices=None, download=True, flatten=True):
        self.base_dataset = datasets.MNIST(root=root, train=(split == 'train' or split == 'valid'), download=download)
        self.indices = indices if indices is not None else list(range(len(self.base_dataset)))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())
        ])
        self.flatten = flatten

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.base_dataset[real_idx]   # 這裡拿到的是 PIL image
        img = self.transform(img) # (1, 28, 28), binary tensor
        img = img.squeeze(0) # (28, 28)
        img = torch.stack([img, 1 - img], dim=-1)  # (H, W, 2)
        if self.flatten:
            img = img.reshape(img.shape[0] * img.shape[1]).long()  # (H*W, )
        return img, label

# class BinaryMNIST(Dataset):
#     """
#     Binarized MNIST dataset.
#     """
#     data_url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'

#     def __init__(self, root, split, transform=None):
#         super().__init__()
#         self.root = root
#         self.split = split
#         self.transform = transform

#         data_path = os.path.join(self.root, f'binarized_mnist_{split}.amat')
#         if not os.path.exists(self.root):
#             os.makedirs(self.root, exist_ok=True)
#         if not os.path.exists(data_path):
#             print(f'Downloading {split} set...')
#             urlretrieve(self.data_url.format(split), data_path)
#         self.data = torch.from_numpy(np.loadtxt(data_path).astype(np.float32))

#     def __getitem__(self, index):
#         x = self.data[index]
#         x = torch.stack([x, 1 - x], dim=-1)
#         if self.transform is not None:
#             x = self.transform(x)
#         return (x,)

#     def __len__(self):
#         return self.data.size(0)
