import cv2
import numpy as np
from torch.utils.data import Dataset


class CustomDS(Dataset):
    def __init__(self, x_set, y_set, resize_shape=None):
        self.resize_shape = resize_shape
        self.x = x_set
        self.y = y_set

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
