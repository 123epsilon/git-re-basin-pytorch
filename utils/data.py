from typing import List
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import numpy as np

"""
dataset: should be a dataset object with two attributes - dataset.targets and dataset.data
dataset.targets should contain the labels (i.e. the Ys) and dataset.data should contain the input
data (i.e. the Xs). Both should be arrays of arrays, this follows the same schema as a 
torchvision.dataset

label_splits: should be a list of lists of integers corresponding to
the labels to be present in each datasplit.

batch_size: batch size argument for constructing the data loaders

kwargs: arbitrary kwargs for constructing the data loaders

Returns an array of dataloaders of length len(label_splits),
each dataloader has only items from the dataset with the set of labels
defined in the list in the corresponding index of label_splits. This function
is useful for splitting datasets for split training.

For example we might load CIFAR10 which has labels 0-9 and split it as follows:
```
cifar10 = torchvision.datasets.CIFAR10(...)
label_splits = [[0,1,2,3,4], [5,6,7,8,9]]
split_dataloaders = get_split_dataset(cifar10, label_splits)
``` 
"""
def get_split_dataset(dataset, label_splits: List[List[int]], batch_size: int, **kwargs) -> List[DataLoader]:
    split_dataloaders = []
    for split in label_splits:
        split_indices = np.where(np.isin(np.array(dataset.targets), split))
        split_dataset = torch.utils.data.Subset(dataset, split_indices[0])
        loader = DataLoader(
            split_dataset, batch_size=batch_size, **kwargs
        )
        split_dataloaders.append(loader)

    return split_dataloaders


"""
Returns train and test datasets for MNIST
"""
def get_mnist():
    transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
      ])
    trainset = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
    testset = datasets.MNIST('../data', train=False,
                      transform=transform)
    
    return trainset, testset

"""
Returns train and test datasets for CIFAR10
"""
def get_cifar10():
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    train_transforms = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transforms)
    
    testset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    
    return trainset, testset

"""
Returns train and test datasets for CIFAR10
"""
def get_cifar100():
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    train_transforms = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=train_transforms)
    
    testset = datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=test_transform)
    
    return trainset, testset

"""
Returns train and test datasets for ImageNet
"""
def get_imagenet():
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    test_transform = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
    
    train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    trainset = datasets.ImageNet(root='./data', train=True,
                                            download=True, transform=train_transforms)
    
    testset = datasets.ImageNet(root='./data', train=False,
                                        download=True, transform=test_transform)
    
    return trainset, testset