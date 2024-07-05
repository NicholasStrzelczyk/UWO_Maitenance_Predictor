import os
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def data_to_xy(data, seperator=" "):
    x_set, y_set = [], []
    for line in data:
        x, y = line.split(seperator)
        x_set.append(x.strip())
        y_set.append(y.strip())
    return x_set, y_set


def get_data_from_list(list_path, split=None):
    all_data = []
    for line in open(list_path, "r"):
        all_data.append(line)

    x1, y1, x2, y2 = None, None, None, None

    if split is not None:
        assert (0.01 <= split <= 0.99)
        data_p1, data_p2 = train_test_split(all_data, test_size=split, random_state=42, shuffle=True)
        x1, y1 = data_to_xy(data_p1, seperator=" ")
        x2, y2 = data_to_xy(data_p2, seperator=" ")
    else:
        x1, y1 = data_to_xy(all_data, seperator=" ")

    return x1, y1, x2, y2


def estimate_class_weight(y_set, resize_shape):
    ratio_list = []
    for path in tqdm(np.unique(y_set), desc='estimating class weights'):
        tgt = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), resize_shape)
        tgt = cv2.normalize(tgt, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        tgt[tgt >= 0.5] = 1
        tgt[tgt < 0.5] = 0
        _, count = np.unique(tgt, return_counts=True)
        ratio_list.append(float(count[1] / count[0]))
    factor = np.round(np.mean(ratio_list), decimals=4)
    return 1.0 - factor


def get_os_dependent_paths(model_ver, partition):
    assert (partition == 'train' or partition == 'test')
    if sys.platform == 'darwin':  # mac
        data_dir = "/Users/nick_1/Bell_5G_Data/1080_snapshots"
        list_path = os.path.join(data_dir, "{}/list_{}.txt".format(partition, sys.platform))
        save_path = "./model_{}".format(model_ver)
    else:  # windows
        data_dir = "C:\\Users\\NickS\\UWO_Summer_Research\\Bell_5G_Data\\1080_snapshots"
        list_path = os.path.join(data_dir, "{}\\list_{}.txt".format(partition, sys.platform))
        save_path = ".\\model_{}".format(model_ver)
    return list_path, save_path


def print_metric_plots(metrics_history, model_ver, save_path):
    for name, m_train, m_val in metrics_history:
        plt.clf()
        plt.plot(m_train)
        plt.plot(m_val)
        plt.title("Training {}".format(name))
        plt.ylabel(name)
        plt.xlabel("epoch")
        plt.legend(['train', 'val'])
        plt.savefig(os.path.join(save_path, 'model_{}_train_{}_plot.png'.format(model_ver, name)))
