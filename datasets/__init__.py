from ._base import get_dataset, register_dataset
from .bmnist import BinaryMNIST
from .text8 import Text8Dataset
from .toy_dfm import ToyDataset

try:
    from .promoter import TSSDatasetS
except ImportError:
    print('[WARNING]: dependencies for TSSDatasetS not installed, skipping import')
