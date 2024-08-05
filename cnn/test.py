import os
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_f1_score, binary_jaccard_index
from tqdm import tqdm

from utils.misc_util import get_os_dependent_paths
from utils.data_import_util import get_data_from_list
from utils.log_util import setup_basic_logger, log_and_print, print_hyperparams
from utils.seed_util import make_deterministic
from custom_ds import CustomDS
from unet_model import UNet


def print_prediction(best_pred, worst_pred, metric_name, metric_value_best, metric_value_worst):
    global model_version, save_path
    f, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))
    axs[0].imshow(best_pred, cmap='gray')
    axs[0].set_title('Best {}={:.4f}'.format(metric_name, metric_value_best))
    axs[1].imshow(worst_pred, cmap='gray')
    axs[1].set_title('Worst {}={:.4f}'.format(metric_name, metric_value_worst))
    plt.suptitle('Model_{} Predictions ({})'.format(model_version, metric_name), fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(save_path, 'model_{}_predictions_{}.png'.format(model_version, metric_name)))


def test(model, loss_fn, test_loader, device):
    global model_version, save_path

    test_loss, test_f1, test_jac = 0.0, 0.0, 0.0
    best_f1, worst_f1 = 0.0, 1.0
    best_jac, worst_jac = 0.0, 1.0

    model.eval()
    log_and_print("{} starting testing...".format(datetime.now()))

    # --- validation step --- #
    with torch.no_grad():
        for image, target in tqdm(test_loader, desc="test progress"):
            image = image.to(device=device)
            target = target.to(device=device)
            output = model(image)
            loss = loss_fn(output, target).item()
            f1_score = binary_f1_score(output, target, threshold=0.5).item()
            jac_idx = binary_jaccard_index(output, target, threshold=0.5).item()

            if f1_score > best_f1:
                best_prediction_f1 = np.copy(np.squeeze(output.detach().cpu().numpy()))
                best_f1 = f1_score
            if f1_score < worst_f1:
                worst_prediction_f1 = np.copy(np.squeeze(output.detach().cpu().numpy()))
                worst_f1 = f1_score

            if jac_idx > best_jac:
                best_prediction_jac = np.copy(np.squeeze(output.detach().cpu().numpy()))
                best_f1 = f1_score
            if jac_idx < worst_jac:
                worst_prediction_jac = np.copy(np.squeeze(output.detach().cpu().numpy()))
                worst_f1 = f1_score

            test_loss += loss
            test_f1 += f1_score
            test_jac += jac_idx
            del image, target, output

    # --- print epoch results --- #
    log_and_print("{} testing metrics:".format(datetime.now()))
    log_and_print("\tloss: {:.9f}, f1_score: {:.9f}, jaccard_idx: {:.9f}".format(
        test_loss / len(test_loader), test_f1 / len(test_loader), test_jac / len(test_loader)))

    # --- save example outputs --- #
    log_and_print("{} generating prediction samples...".format(datetime.now()))
    print_prediction(best_prediction_f1, worst_prediction_f1, 'f1_score', best_f1, worst_f1)
    print_prediction(best_prediction_jac, worst_prediction_jac, 'jaccard_index', best_jac, worst_jac)
    log_and_print("{} testing complete.".format(datetime.now()))


if __name__ == '__main__':
    # hyperparameters
    model_version = 3
    resize_shape = (512, 512)
    loss_fn_name = 'binary_cross_entropy'
    seed = 222333444  # same seed as training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    list_path, save_path = get_os_dependent_paths(model_version, partition='test')
    weights_file = os.path.join(save_path, "model_{}_weights.pth".format(model_version))

    # set up logger and deterministic seed
    setup_basic_logger(os.path.join(save_path, 'testing.log'))
    make_deterministic(seed)

    # print training hyperparameters
    print_hyperparams(
        model_ver=model_version, resize_shape=resize_shape, loss_fn_name=loss_fn_name, seed=seed, device=device
    )

    # set up dataset(s)
    x_test, y_test, _, _ = get_data_from_list(list_path, split=None)
    test_ds = CustomDS(x_test, y_test, resize_shape=resize_shape)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # compile model
    model = UNet()
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.to(device=device)

    # init model training parameters
    loss_fn = torch.nn.BCELoss()

    # test model
    test(model, loss_fn, test_loader, device)
