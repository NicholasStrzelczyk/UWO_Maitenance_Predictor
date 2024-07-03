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
from custom_model import UNet
from utils.data_helper import get_data_from_list, get_os_dependent_paths


def test(model, loss_fn, test_loader, device):
    global model_version, save_path

    precision = BinaryPrecision(threshold=0.5).to(device=device)
    recall = BinaryRecall(threshold=0.5).to(device=device)
    f1_score = BinaryF1Score(threshold=0.5).to(device=device)

    test_loss, test_bp, test_br, test_bf1 = 0.0, 0.0, 0.0, 0.0
    predictions = []

    model.eval()
    print("{} starting testing for model {}...".format(datetime.now(), model_version))

    # --- validation step --- #
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="test progress"):
            images = images.to(device=device)
            targets = targets.to(device=device)
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()
            test_bp += precision(outputs, targets).item()
            test_br += recall(outputs, targets).item()
            test_bf1 += f1_score(outputs, targets).item()
            for pred in outputs:
                pred = np.squeeze(pred.detach().cpu().numpy())
                predictions.append(pred)
            del images, targets, outputs

    # --- print epoch results --- #
    print("{} testing metrics:".format(datetime.now()))
    print("\tloss: {:.9f}, precision: {:.9f}, recall: {:.9f}, f1_score: {:.9f}".format(
        test_loss / len(test_loader), test_bp / len(test_loader),
        test_br / len(test_loader), test_bf1 / len(test_loader)))

    # --- save example outputs --- #
    f, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    axs[0, 0].imshow(predictions[0], cmap='gray')
    axs[0, 1].imshow(predictions[1], cmap='gray')
    axs[1, 0].imshow(predictions[2], cmap='gray')
    axs[1, 1].imshow(predictions[3], cmap='gray')
    plt.suptitle('Predictions', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(save_path, 'model_{}_predictions.png'.format(model_version)))


if __name__ == '__main__':
    # hyperparameters
    model_version = 1
    batch_sz = 2  # batch size (2 works best on gpu)
    resize_shape = (512, 512)
    list_path, save_path = get_os_dependent_paths(model_version, partition='test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up dataset(s)
    x_test, y_test, _, _ = get_data_from_list(list_path, split=None)
    test_ds = BellGrayDS(x_test, y_test, resize_shape=resize_shape)
    test_loader = DataLoader(test_ds, batch_size=batch_sz, shuffle=False)

    # compile model
    model = UNet(n_channels=1, n_classes=1)
    torch.load(os.path.join(save_path, "model_{}_weights.pth".format(model_version)))
    model.to(device=device)

    # init model training parameters
    # class_weight_alpha = estimate_class_weight(y_test, resize_shape=resize_shape)
    # print("Class weight alpha: {}".format(class_weight_alpha))
    class_weight_alpha = 0.75
    loss_fn = FocalBCELoss(alpha=class_weight_alpha, gamma=2.0)

    # test model
    test(model, loss_fn, test_loader, device)
