import os
from datetime import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from torchsummary import summary
from tqdm import tqdm

from custom_ds import BellGrayDS
from custom_loss import FocalBCELoss
from custom_model import UNet
from utils.data_helper import get_os_dependent_paths, get_data_from_list, print_metric_plots


def train(model, loss_fn, optimizer, scheduler, train_loader, val_loader, n_epochs, device):
    global model_version, save_path

    precision = BinaryPrecision(threshold=0.5).to(device=device)
    recall = BinaryRecall(threshold=0.5).to(device=device)
    f1_score = BinaryF1Score(threshold=0.5).to(device=device)

    losses_train, losses_val = [], []
    precision_train, precision_val = [], []
    recall_train, recall_val = [], []
    f1_train, f1_val = [], []

    model.train()
    print("{} starting training for model {}...".format(datetime.now(), model_version))

    # --- iterate through all epochs --- #
    for epoch in range(n_epochs):
        train_loss, val_loss = 0.0, 0.0
        train_bp, val_bp = 0.0, 0.0
        train_br, val_br = 0.0, 0.0
        train_bf1, val_bf1 = 0.0, 0.0

        # --- training step --- #
        for images, targets in tqdm(train_loader, desc="epoch {} train progress".format(epoch + 1)):
            images = images.to(device=device)
            targets = targets.to(device=device)
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bp += precision(outputs, targets).item()
            train_br += recall(outputs, targets).item()
            train_bf1 += f1_score(outputs, targets).item()
            del images, targets, outputs

        losses_train.append(train_loss / len(train_loader))
        precision_train.append(train_bp / len(train_loader))
        recall_train.append(train_br / len(train_loader))
        f1_train.append(train_bf1 / len(train_loader))

        # --- validation step --- #
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="epoch {} val progress".format(epoch + 1)):
                images = images.to(device=device)
                targets = targets.to(device=device)
                outputs = model(images)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()
                val_bp += precision(outputs, targets).item()
                val_br += recall(outputs, targets).item()
                val_bf1 += f1_score(outputs, targets).item()
                del images, targets, outputs

        scheduler.step(val_loss)

        losses_val.append(val_loss / len(val_loader))
        precision_val.append(val_bp / len(val_loader))
        recall_val.append(val_br / len(val_loader))
        f1_val.append(val_bf1 / len(val_loader))

        # --- print epoch results --- #
        print("{} epoch {}/{} metrics:".format(datetime.now(), epoch + 1, n_epochs))
        print("\t[train] loss: {:.9f}, precision: {:.9f}, recall: {:.9f}, f1_score: {:.9f}".format(
            losses_train[epoch], precision_train[epoch], recall_train[epoch], f1_train[epoch]))
        print("\t[valid] loss: {:.9f}, precision: {:.9f}, recall: {:.9f}, f1_score: {:.9f}".format(
            losses_val[epoch], precision_val[epoch], recall_val[epoch], f1_val[epoch]))

    # --- save weights and plot metrics --- #
    torch.save(model.state_dict(), os.path.join(save_path, "model_{}_weights.pth".format(model_version)))
    metrics_history = [
        ("loss", losses_train, losses_val),
        ("precision", precision_train, precision_val),
        ("recall", recall_train, recall_val),
        ("f1_score", f1_train, f1_val),
    ]
    print_metric_plots(metrics_history, model_version, save_path)


if __name__ == '__main__':
    # hyperparameters
    model_version = 1
    n_epochs = 20  # num of epochs
    batch_sz = 2  # batch size (2 works best on gpu)
    lr = 0.0001  # learning rate
    wd = 0.00001  # weight decay
    resize_shape = (512, 512)
    list_path, save_path = get_os_dependent_paths(model_version, partition='train')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up dataset(s)
    x_train, y_train, x_val, y_val = get_data_from_list(list_path, split=0.2)
    train_ds = BellGrayDS(x_train, y_train, resize_shape=resize_shape)
    val_ds = BellGrayDS(x_val, y_val, resize_shape=resize_shape)
    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_sz, shuffle=False)

    # compile model
    model = UNet(n_channels=1, n_classes=1)
    model.to(device=device)

    # init model training parameters
    # class_weight_alpha = estimate_class_weight(y_train, resize_shape=resize_shape)
    # print("Class weight alpha: {}".format(class_weight_alpha))
    class_weight_alpha = 0.75
    loss_fn = FocalBCELoss(alpha=class_weight_alpha, gamma=2.0)
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    # run torch summary report
    summary(model, input_size=(1, 512, 512))

    # train model
    train(model, loss_fn, optimizer, scheduler, train_loader, val_loader, n_epochs, device)
