import csv
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.functional.classification import binary_f1_score, binary_jaccard_index
from tqdm import tqdm

from utils.data_import_util import get_xy_data
from utils.log_util import setup_basic_logger, log_and_print, print_hyperparams
from custom_ds import CustomDS
from unet_model import UNet


def plot_metric(metric, metric_name):
    global model_version, save_path
    plt.clf()
    metric.plot(score=True)
    plt.savefig(os.path.join(save_path, 'model_{}_test_{}.png'.format(model_version, metric_name)))


def print_hist(metric_vals, metric_name):
    global model_version, save_path
    plt.clf()
    plt.figure(figsize=(8, 6))
    values, bins, bars = plt.hist(metric_vals, edgecolor='white')
    plt.xlabel(metric_name)
    plt.ylabel('number of predictions')
    plt.bar_label(bars)
    plt.suptitle('Prediction Histogram for {}'.format(metric_name), fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(save_path, 'model_{}_test_{}.png'.format(model_version, metric_name)))


def test(model, test_loader, device):
    global model_version, save_path, jac_ex_save_path, f1_ex_save_path

    jac_pred_count = 1
    f1_pred_count = 1

    metrics_csv_list = []
    f1_scores, jac_idxs = [], []
    bprc = BinaryPrecisionRecallCurve()
    model.eval()
    log_and_print("{} starting testing...".format(datetime.now()))

    # --- validation step --- #
    with torch.no_grad():
        for image, target in tqdm(test_loader, desc="test progress"):
            image = image.to(device=device)
            target = target.to(device=device)
            output = model(image)

            f1 = binary_f1_score(output, target).item()
            jac = binary_jaccard_index(output, target).item()
            bprc.update(output, target.long())

            f1_scores.append(f1)
            jac_idxs.append(jac)
            metrics_csv_list.append([f1, jac])

            if jac == 0.0:
                cv2.imwrite(
                    os.path.join(jac_ex_save_path, 'jac_pred_{}.png'.format(jac_pred_count)),
                    255 * np.squeeze(output.detach().cpu().numpy())
                )
                cv2.imwrite(
                    os.path.join(jac_ex_save_path, 'jac_targ_{}.png'.format(jac_pred_count)),
                    255 * np.squeeze(target.detach().cpu().numpy())
                )
                jac_pred_count += 1

            if f1 == 0.0:
                cv2.imwrite(
                    os.path.join(f1_ex_save_path, 'f1_pred_{}.png'.format(f1_pred_count)),
                    255 * np.squeeze(output.detach().cpu().numpy())
                )
                cv2.imwrite(
                    os.path.join(f1_ex_save_path, 'f1_targ_{}.png'.format(f1_pred_count)),
                    255 * np.squeeze(target.detach().cpu().numpy())
                )
                f1_pred_count += 1

            del image, target, output

    # --- print epoch results --- #
    log_and_print("{} testing metrics:".format(datetime.now()))
    log_and_print("\tf1_score:\t{:.9f} (best), {:.9f} (worst), {:.9f} (avg)".format(
        np.max(f1_scores), np.min(f1_scores), np.mean(f1_scores)))
    log_and_print("\tjaccard_idx:\t{:.9f} (best), {:.9f} (worst), {:.9f} (avg)".format(
        np.max(jac_idxs), np.min(jac_idxs), np.mean(jac_idxs)))

    # --- save metric outputs --- #
    log_and_print("{} generating prediction plots and figures...".format(datetime.now()))
    print_hist(f1_scores, 'f1_score')
    print_hist(jac_idxs, 'jaccard_index')
    plot_metric(bprc, 'prc')

    csv_path = os.path.join(save_path, 'predictions.csv')
    open(csv_path, 'w+').close()  # overwrite/ make new blank file
    with open(csv_path, 'a', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['f1_score', 'jaccard_index'])
        writer.writerows(metrics_csv_list)

    log_and_print("{} testing complete.".format(datetime.now()))


if __name__ == '__main__':
    # hyperparameters
    model_version = 1
    input_shape = (512, 512)
    dataset_name = 'synth_datasets'
    weights_filename = 'e50_weights.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set up paths and directories
    save_path = os.path.join('.', 'model_{}'.format(model_version))
    jac_ex_save_path = os.path.join(save_path, 'pred_examples', 'jac')
    f1_ex_save_path = os.path.join(save_path, 'pred_examples', 'f1')
    os.makedirs(jac_ex_save_path, exist_ok=True)
    os.makedirs(f1_ex_save_path, exist_ok=True)

    # set up logger and deterministic seed
    setup_basic_logger(os.path.join(save_path, 'testing.log'))

    # print training hyperparameters
    print_hyperparams(
        model_ver=model_version, input_shape=input_shape, dataset_name=dataset_name,
        weights_filename=weights_filename, device=device
    )

    # set up dataset(s)
    x_test, y_test, _, _ = get_xy_data(dataset_name, partition='test')
    test_ds = CustomDS(x_test, y_test, resize_shape=input_shape)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # compile model
    model = UNet()
    weights_file = os.path.join(save_path, 'weights', weights_filename)
    model.load_state_dict(torch.load(weights_file, map_location=device, weights_only=True))
    model.to(device=device)

    # test model
    test(model, test_loader, device)
