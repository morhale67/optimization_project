from torchvision import datasets
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import Subset


def load_data_original(batch_size=128, num_workers=0):
    """
    num_workers - number of subprocesses to use for data loading.
    batch_size - number of samples per batch to load.
    """
    transform = transforms.ToTensor()  # convert data to torch.FloatTensor

    # choose the training and test datasets
    train_data = datasets.CIFAR100(root='data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR100(root='data', train=False, download=True, transform=transform)

    # slice the data
    num_train_samples = 128*100
    # train_data = Subset(train_data, np.arange(num_train_samples))
    num_train_samples = 128*20
    # test_data = Subset(test_data, np.arange(num_train_samples))

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    return train_loader, test_loader, batch_size

