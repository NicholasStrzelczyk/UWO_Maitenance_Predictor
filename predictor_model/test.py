import os
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from tqdm import tqdm

from custom_ds import BellGrayDS
from custom_loss import FocalBCELoss
from custom_model import UNetGrayscale
from utils.data_helper import get_data_from_list, get_os_dependent_paths


def test(model, loss_fn, test_loader, device):
    global model_version, save_path

    precision = BinaryPrecision(threshold=0.5).to(device=device)
    recall = BinaryRecall(threshold=0.5).to(device=device)
    f1_score = BinaryF1Score(threshold=0.5).to(device=device)

    test_loss, test_bp, test_br, test_bf1 = 0.0, 0.0, 0.0, 0.0
    best_f1 = 0.0
    worst_f1 = 1.0

    model.eval()
    print("{} starting testing for model {}...".format(datetime.now(), model_version))

    # --- validation step --- #
    with torch.no_grad():
        for image, target in tqdm(test_loader, desc="test progress"):
            image = image.to(device=device)
            target = target.to(device=device)
            output = model(image)
            loss = loss_fn(output, target)
            test_loss += loss.item()
            test_bp += precision(output, target).item()
            test_br += recall(output, target).item()
            test_bf1 += f1_score(output, target).item()

            current_f1 = f1_score(output, target).item()
            if current_f1 > best_f1:
                best_prediction = np.copy(np.squeeze(output.detach().cpu().numpy()))
                best_f1 = current_f1

            if current_f1 < worst_f1:
                worst_prediction = np.copy(np.squeeze(output.detach().cpu().numpy()))
                worst_f1 = current_f1

            del image, target, output

    # --- print epoch results --- #
    print("{} testing metrics:".format(datetime.now()))
    print("\tloss: {:.9f}, precision: {:.9f}, recall: {:.9f}, f1_score: {:.9f}".format(
        test_loss / len(test_loader), test_bp / len(test_loader),
        test_br / len(test_loader), test_bf1 / len(test_loader)))

    # --- save example outputs --- #
    f, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))
    axs[0].imshow(best_prediction, cmap='gray')
    axs[0].set_title('Best Prediction (f1_score={:.4f})'.format(best_f1))
    axs[1].imshow(worst_prediction, cmap='gray')
    axs[1].set_title('Worst Prediction (f1_score={:.4f})'.format(worst_f1))
    plt.suptitle('Model_{} Predictions'.format(model_version), fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(save_path, 'model_{}_predictions.png'.format(model_version)))


if __name__ == '__main__':
    # hyperparameters
    model_version = 3
    resize_shape = (512, 512)
    list_path, save_path = get_os_dependent_paths(model_version, partition='test')
    weights_file = os.path.join(save_path, "model_{}_weights.pth".format(model_version))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up dataset(s)
    x_test, y_test, _, _ = get_data_from_list(list_path, split=None)
    test_ds = BellGrayDS(x_test, y_test, resize_shape=resize_shape)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # compile model
    model = UNetGrayscale()
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.to(device=device)

    # init model training parameters
    # class_weight_alpha = estimate_class_weight(y_test, resize_shape=resize_shape)
    # print("Class weight alpha: {}".format(class_weight_alpha))
    class_weight_alpha = 0.75
    loss_fn = FocalBCELoss(alpha=class_weight_alpha, gamma=2.0)

    # test model
    test(model, loss_fn, test_loader, device)
