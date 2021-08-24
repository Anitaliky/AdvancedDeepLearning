import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from PIL import Image

from . import utils

def train_eval_loaders_cifar10(data_dir,
                               batch_size,
                               angles,
                               eval_size=0.2,
                               num_workers=4,
                               augment=False,
                               class_name=None):
    normalize = transforms.Normalize(utils.cifar10_mean, utils.cifar10_std)

    if augment:
        transform = transforms.Compose([
            utils.RotateAngle(angles=angles),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    if class_name:
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True,
        )
        x_train = train_dataset.data
        y_train = train_dataset.targets
        
        train_dataset = DatasetMaker(
            [get_class_i(x_train, y_train, utils.classDict[class_name])],
            transform
        )
        
        eval_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True,
        )
        x_eval = eval_dataset.data
        y_eval = eval_dataset.targets
        
        eval_dataset = DatasetMaker(
            [get_class_i(x_eval, y_eval, utils.classDict[class_name])],
            transform
        )
        
    else:
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=transform,
        )

        eval_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=transform,
        )

    data_size = len(train_dataset)
    indices = list(range(data_size))
    train_size = int(np.floor(eval_size * data_size))

    train_idx, eval_idx = indices[train_size:], indices[:train_size]
    train_sampler = SubsetRandomSampler(train_idx)
    eval_sampler = SubsetRandomSampler(eval_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, drop_last=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, sampler=eval_sampler,
        num_workers=num_workers, drop_last=True
    )

    return train_loader, eval_loader


def test_loader(data_dir,
                batch_size,
                angles,
                class_name=None,
                augment=False,
                num_workers=4,):

    normalize = transforms.Normalize(utils.cifar10_mean, utils.cifar10_std)

    if augment:
        transform = transforms.Compose([
            utils.RotateAngle(angles=angles),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
    if class_name:
        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True,
        )
        x_test = dataset.data
        y_test = dataset.targets
        
        dataset = DatasetMaker(
            [get_class_i(x_test, y_test, utils.classDict[class_name])],
            transform
        )
    else:
        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
        )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        num_workers=num_workers, drop_last=True
    )

    return data_loader

#inspired by https://gist.github.com/Miladiouss/6ba0876f0e2b65d0178be7274f61ad2f
# Define a function to separate CIFAR classes by class index

def get_class_i(x, y, i):
    """
    x: trainset.data or testset.data
    y: trainset.targets or testset.targets
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:,0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]
    
    return x_i

class DatasetMaker(torch.utils.data.Dataset):
    def __init__(self, datasets, transformFunc=None):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths  = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc
        
    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = Image.fromarray(img)
        if self.transformFunc:
            img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)
    
    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index  = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class