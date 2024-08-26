import csv
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_f1_score, binary_jaccard_index
from tqdm import tqdm

from utils.data_import_util import get_xy_data
from custom_ds import CustomDS
from unet_model import UNet


def test(model, test_loader, device):
    global model_version, save_path, pred_ex_save_path

    pred_count = 1
    metrics_csv_list = []
    model.eval()
    print("{} creating testing samples...".format(datetime.now()))

    # --- validation step --- #
    with torch.no_grad():
        for image, target in tqdm(test_loader, desc="test progress"):
            image = image.to(device=device)
            target = target.to(device=device)
            output = model(image)

            f1 = binary_f1_score(output, target).item()
            jac = binary_jaccard_index(output, target).item()

            if f1 <= 0.5 or jac <= 0.5:
                cv2.imwrite(
                    os.path.join(pred_ex_save_path, 'preds', 'pred_{}.png'.format(pred_count)),
                    255 * np.squeeze(output.detach().cpu().numpy())
                )
                cv2.imwrite(
                    os.path.join(pred_ex_save_path, 'targs', 'targ_{}.png'.format(pred_count)),
                    255 * np.squeeze(target.detach().cpu().numpy())
                )
                cv2.imwrite(
                    os.path.join(pred_ex_save_path, 'inputs', 'input_{}.png'.format(pred_count)),
                    255 * np.transpose(np.squeeze(image.detach().cpu().numpy()), axes=(1, 2, 0))
                )
                metrics_csv_list.append([pred_count, f1, jac])
                pred_count += 1

            del image, target, output

    csv_path = os.path.join(pred_ex_save_path, 'prediction_scores.csv')
    open(csv_path, 'w+').close()  # overwrite/ make new blank file
    with open(csv_path, 'a', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['pred_num', 'f1_score', 'jaccard_index'])
        writer.writerows(metrics_csv_list)
    print("{} testing samples complete.".format(datetime.now()))


if __name__ == '__main__':
    # hyperparameters
    model_version = 2
    input_shape = (512, 512)
    # dataset_name = 'synth_datasets'
    dataset_name = 'sm_rand_spots'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set up paths and directories
    save_path = os.path.join('.', 'model_{}'.format(model_version))
    pred_ex_save_path = os.path.join(save_path, 'pred_examples')
    os.makedirs(os.path.join(pred_ex_save_path, 'preds'), exist_ok=True)
    os.makedirs(os.path.join(pred_ex_save_path, 'targs'), exist_ok=True)
    os.makedirs(os.path.join(pred_ex_save_path, 'inputs'), exist_ok=True)

    # print training hyperparameters
    print('Hyperparameters:')
    print('\tmodel_ver: {}'.format(model_version))
    print('\tinput_shape: {}'.format(input_shape))
    print('\tdataset_name: {}'.format(dataset_name))
    print('\tdevice: {}'.format(device))

    # set up dataset(s)
    x_test, y_test, _, _ = get_xy_data(dataset_name, partition='test')
    # test_ds = CustomDS(x_test, y_test, dataset_name, input_shape)
    test_ds = CustomDS(x_test, y_test, dataset_name)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # compile model
    model = UNet()
    weights_file = os.path.join(save_path, 'best_weights.pth')
    model.load_state_dict(torch.load(weights_file, map_location=device, weights_only=True))
    model.to(device=device)

    # test model
    test(model, test_loader, device)
