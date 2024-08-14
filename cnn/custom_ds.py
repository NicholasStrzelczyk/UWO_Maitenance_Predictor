import os
import sys

import cv2
import numpy as np
from torch.utils.data import Dataset

from utils.constants import *
from utils.misc_util import fix_path


class CustomDS(Dataset):  # OLD DATASET FULL SIZE
    def __init__(self, x_set, y_set, resize_shape=None):
        self.resize_shape = resize_shape
        if sys.platform == 'darwin':
            root_dir = os.path.join(data_path_mac, 'synth_datasets')
        elif sys.platform == 'win32':
            root_dir = os.path.join(data_path_win32, 'synth_datasets')
        else:
            root_dir = os.path.join(data_path_linux, 'synth_datasets')
        self.x, self.y = [], []
        for idx in range(len(x_set)):
            self.x.append(os.path.join(root_dir, fix_path(x_set[idx])))
            self.y.append(os.path.join(root_dir, fix_path(y_set[idx])))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = cv2.imread(self.x[idx], cv2.IMREAD_COLOR)
        target = cv2.imread(self.y[idx], cv2.IMREAD_GRAYSCALE)

        if self.resize_shape is not None:
            image = cv2.resize(image, self.resize_shape)
            target = cv2.resize(target, self.resize_shape)

        image = np.transpose(image, axes=(2, 0, 1))  # transpose RGB image to be (C, H, W) instead of (H, W, C)
        target = np.expand_dims(target, axis=0)  # add dimension for where channels normally are

        # normalize values from [0, 255] to [0, 1]
        image = image.astype(np.float32)
        target = target.astype(np.float32)
        image *= (1 / 255.0)
        target *= (1 / 255.0)

        # ensure that the target is still binary after previous operations
        target[target < 0.5] = 0
        target[target >= 0.5] = 1

        return image, target


class RandSpotsDS(Dataset):  # FULL SIZE RANDOM SPOTS w/ RESIZING
    def __init__(self, x_set, y_set, resize_shape=None):
        if sys.platform == 'darwin':
            root_dir = os.path.join(data_path_mac, 'rand_spots_ds')
        elif sys.platform == 'win32':
            root_dir = os.path.join(data_path_win32, 'rand_spots_ds')
        else:
            root_dir = os.path.join(data_path_linux, 'rand_spots_ds')

        self.resize_shape = resize_shape
        self.x, self.y = [], []
        for idx in range(len(x_set)):
            self.x.append(os.path.join(root_dir, fix_path(x_set[idx])))
            self.y.append(os.path.join(root_dir, fix_path(y_set[idx])))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = cv2.imread(self.x[idx], cv2.IMREAD_COLOR)
        target = cv2.imread(self.y[idx], cv2.IMREAD_GRAYSCALE)

        if self.resize_shape is not None:
            image = cv2.resize(image, self.resize_shape)
            target = cv2.resize(target, self.resize_shape)

        image = np.transpose(image, axes=(2, 0, 1))  # transpose RGB image to be (C, H, W) instead of (H, W, C)
        target = np.expand_dims(target, axis=0)  # add dimension for where channels normally are

        # normalize values from [0, 255] to [0, 1]
        image = image.astype(np.float32)
        target = target.astype(np.float32)
        image *= (1 / 255.0)
        target *= (1 / 255.0)

        # ensure that the target is still binary after previous operations
        target[target < 0.5] = 0
        target[target >= 0.5] = 1

        return image, target


class SmRandSpotsDS(Dataset):  # PREPROCESSED RANDOM SPOTS (no resizing)
    def __init__(self, x_set, y_set):
        if sys.platform == 'darwin':
            root_dir = os.path.join(data_path_mac, 'sm_rand_spots')
        elif sys.platform == 'win32':
            root_dir = os.path.join(data_path_win32, 'sm_rand_spots')
        else:
            root_dir = os.path.join(data_path_linux, 'sm_rand_spots')
        self.x, self.y = [], []
        for idx in range(len(x_set)):
            self.x.append(os.path.join(root_dir, fix_path(x_set[idx])))
            self.y.append(os.path.join(root_dir, fix_path(y_set[idx])))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = cv2.imread(self.x[idx], cv2.IMREAD_COLOR)
        target = cv2.imread(self.y[idx], cv2.IMREAD_GRAYSCALE)

        image = np.transpose(image, axes=(2, 0, 1))  # transpose RGB image to be (C, H, W) instead of (H, W, C)
        target = np.expand_dims(target, axis=0)  # add dimension for where channels normally are

        # normalize values from [0, 255] to [0, 1]
        image = image.astype(np.float32)
        target = target.astype(np.float32)
        image *= (1 / 255.0)
        target *= (1 / 255.0)

        # ensure that the target is still binary after previous operations
        target[target < 0.5] = 0
        target[target >= 0.5] = 1

        return image, target
