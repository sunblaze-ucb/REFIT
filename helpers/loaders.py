import os

import numpy as np
import torch

import torchvision
import torchvision.transforms as transforms
import datasets.cifar_partial
import datasets.stl_partial
import datasets.imagenet32
from helpers.consts import *
from helpers.ImageFolderCustomClass import ImageFolderCustomClass

def _getdatatransformsdb(datatype):
    transform_train, transform_test = None, None
    if datatype.lower() == CIFAR10 or datatype.lower() == CIFAR100:
        # Data preperation
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    if datatype.lower() == 'imagenet32':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4811, 0.4575, 0.4079), (0.2604, 0.2532, 0.2682)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4811, 0.4575, 0.4079), (0.2604, 0.2532, 0.2682)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    return transform_train, transform_test


def getdataloader(datatype, train_db_path, test_db_path, batch_size, workers = 4):
    # get transformations
    transform_train, transform_test = _getdatatransformsdb(datatype=datatype.lower().split('+')[0].split('_')[0])
    n_classes = 0

    # Data loaders
    
    if datatype.lower().split('+')[0] == 'imagenet32':
        print("Using resized imagenet(32 * 32)")
        tmp = datatype.split('+')
        lbl_range = (int(tmp[1]), int(tmp[2])) ##[0, 1000)
        id_range = (int(tmp[3]), int(tmp[4])) #[1,11)
        n_classes = lbl_range[1] - lbl_range[0]
        trainset = datasets.imagenet32.IMAGENET32(train=True, root = train_db_path, lbl_range = lbl_range,
                                                id_range = id_range,
                                                transform = transform_train)
        if n_classes > 1000:
            testset = None
        else:
            testset = datasets.imagenet32.IMAGENET32(train=False, root = train_db_path, lbl_range = lbl_range,
                                                id_range = id_range,
                                                transform = transform_test)
        
    elif datatype.lower() == CIFAR10:
        print("Using CIFAR10 dataset.")
        trainset = torchvision.datasets.CIFAR10(root=train_db_path,
                                                train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=test_db_path,
                                               train=False, download=True,
                                               transform=transform_test)
        n_classes = 10
    elif datatype.lower() == CIFAR100:
        print("Using CIFAR100 dataset.")
        trainset = torchvision.datasets.CIFAR100(root=train_db_path,
                                                 train=True, download=True,
                                                 transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=test_db_path,
                                                train=False, download=True,
                                                transform=transform_test)
        n_classes = 100
    elif datatype.lower() == STL:
        print("Using STL10 dataset.")
        trainset = torchvision.datasets.STL10(root=train_db_path,
                                                 split='train', download=True,
                                                 transform=transform_train)
        testset = torchvision.datasets.STL10(root=test_db_path,
                                                split='test', download=True,
                                                transform=transform_test)
        n_classes = 10
    elif datatype.lower().split('+')[0] == CIFAR10 + '_partial':
        print("Using CIFAR10 dataset partially.")
        trainset = datasets.cifar_partial.CIFAR10(root=train_db_path,
                                                train=True, download=True,
                                                transform=transform_train, ratio = float(datatype.split('+')[1]),
                                                remain_size = len(datatype.split('+')) == 2)
        testset = datasets.cifar_partial.CIFAR10(root=test_db_path,
                                               train=False, download=True,
                                               transform=transform_test)
        n_classes = 10
    elif datatype.lower().split('+')[0] == CIFAR100 + '_partial':
        print("Using CIFAR100 dataset partially.")
        trainset = datasets.cifar_partial.CIFAR100(root=train_db_path,
                                                train=True, download=True,
                                                transform=transform_train, ratio = float(datatype.split('+')[1]),
                                                remain_size = len(datatype.split('+')) == 2)
        testset = datasets.cifar_partial.CIFAR100(root=test_db_path,
                                               train=False, download=True,
                                               transform=transform_test)
        n_classes = 100
    elif datatype.lower().split('+')[0] == STL + '_unlabeled' + '_partial':
        print("Using the unlabeled part of STL10 dataset partially.")
        trainset = datasets.stl_partial.STL10(root=train_db_path,
                                                 split='unlabeled', download=True,
                                                 transform=transform_train, ratio = float(datatype.split('+')[1]),
                                                 remain_size = len(datatype.split('+')) == 2)
        testset = None
        n_classes = 10
    elif datatype.lower() == STL + '_unlabeled':
        print("Using the unlabeled part of STL10 dataset.")
        trainset = torchvision.datasets.STL10(root=train_db_path,
                                                 split='unlabeled', download=True,
                                                 transform=transform_train)
        testset = None
        n_classes = 10
    else:
        print("Dataset is not supported.")
        return None, None, None

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=workers)
    if testset != None:
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=workers)
    else:
        testloader = None
    return trainloader, testloader, n_classes


def _getdatatransformswm(is_imgnet32 = False):
    if is_imgnet32:
        transform_wm = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4811, 0.4575, 0.4079), (0.2604, 0.2532, 0.2682)),
        ])
    else:
        transform_wm = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    return transform_wm


def getwmloader(wm_path, batch_size, labels_path, shuffle = True):
    if len(wm_path.split('+')) == 2:
        is_imgnet32 = wm_path.split('+')[0] == 'imgnet'
        wm_path = wm_path.split('+')[1]
    else:
        is_imgnet32 = False
    
    transform_wm = _getdatatransformswm(is_imgnet32)
    
    # load watermark images
    wmloader = None

    wmset = ImageFolderCustomClass(
        wm_path,
        transform_wm)
    img_nlbl = []
    wm_targets = np.loadtxt(os.path.join(wm_path, labels_path))
    for idx, (path, target) in enumerate(wmset.imgs):
        img_nlbl.append((path, int(wm_targets[idx])))
    wmset.imgs = img_nlbl

    wmloader = torch.utils.data.DataLoader(
        wmset, batch_size=batch_size, shuffle=shuffle,
        num_workers=4, pin_memory=True)

    return wmloader

def batch_gen(loader):
    while True:
        for batch in loader:
            yield batch
