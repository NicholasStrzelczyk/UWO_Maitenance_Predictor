import os
import sys

from matplotlib import pyplot as plt

from utils.constants import *


def fix_path(path):
    return path.replace('/', '\\') if sys.platform == 'win32' else path


def get_list_path(partition, ds_parent_folder):
    assert (ds_parent_folder in valid_ds_folders)
    assert (partition == 'train' or partition == 'test' or partition == 'validation')
    if sys.platform == 'darwin':  # mac
        result = os.path.join(data_path_mac, ds_parent_folder, partition, 'list.txt')
    elif sys.platform == 'win32':  # windows
        result = os.path.join(data_path_win32, ds_parent_folder, partition, 'list.txt')
    else:  # ubuntu
        result = os.path.join(data_path_linux, ds_parent_folder, partition, 'list.txt')
    return result


def get_os_dependent_paths(model_ver, partition):
    assert (partition == 'train' or partition == 'test' or partition == 'validation')
    if sys.platform == 'darwin':  # mac
        data_dir = '/Users/nick_1/Bell_5G_Data/synth_datasets'
        list_path = os.path.join(data_dir, '{}/list_{}.txt'.format(partition, sys.platform))
        save_path = './model_{}'.format(model_ver)
    elif sys.platform == 'win32':  # windows
        data_dir = 'C:\\Users\\NickS\\UWO_Summer_Research\\Bell_5G_Data\\synth_datasets'
        list_path = os.path.join(data_dir, '{}\\list_{}.txt'.format(partition, sys.platform))
        save_path = '.\\model_{}'.format(model_ver)
    else:  # ubuntu
        data_dir = '/mnt/storage_1/bell_5g_datasets/synth_datasets'
        list_path = os.path.join(data_dir, '{}/list_{}.txt'.format(partition, sys.platform))
        save_path = './model_{}'.format(model_ver)

    # make the folder if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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
