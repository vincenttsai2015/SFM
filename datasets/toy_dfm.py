import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from ._base import register_dataset

@register_dataset('toy_dfm')
class ToyDataset(IterableDataset):
    """
    Adapted from `https://github.com/HannesStark/dirichlet-flow-matching/blob/main/utils/dataset.py`.
    """
    def __init__(self, root, split, manifold, probs: Tensor, toy_seq_len: int, toy_simplex_dim: int, sz: int = 100_000):
        super().__init__()
        self.root = root
        self.split = split
        self.m = manifold
        self.sz = sz
        self.seq_len = toy_seq_len
        self.alphabet_size = toy_simplex_dim
        self.probs = torch.as_tensor(probs, dtype=torch.float)
        self.probs = self.probs / self.probs.sum()

    def __len__(self) -> int:
        return self.sz

    def __iter__(self):
        while True:
            sample = torch.multinomial(replacement=True, num_samples=self.seq_len, input=self.probs).squeeze()
            one_hot = nn.functional.one_hot(sample, self.alphabet_size).float()
            # if there is a need to smooth labels, it is done in the model's training step
            yield one_hot.reshape((self.seq_len, self.alphabet_size))
