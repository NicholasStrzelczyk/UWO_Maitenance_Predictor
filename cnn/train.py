import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import dice
from torchmetrics.functional.classification import binary_f1_score, binary_jaccard_index
from torchsummary import summary
from tqdm import tqdm

from utils.data_import_util import get_data_from_list
from utils.misc_util import print_metric_plots, get_os_dependent_paths
from utils.log_util import log_and_print, setup_basic_logger, print_hyperparams
from utils.seed_util import get_random_seed, make_deterministic
from custom_ds import CustomDS
from unet_model import UNet


def train(model, loss_fn, optimizer, scheduler, train_loader, val_loader, n_epochs, device):
    global model_name, model_version, save_path

    losses_train, losses_val = [], []
    f1_train, f1_val = [], []
    jaccard_train, jaccard_val = [], []
    dice_train, dice_val = [], []

    # --- iterate through all epochs --- #
    log_and_print("{} starting training...".format(datetime.now()))
    for epoch in range(n_epochs):

        # --- training step --- #
        model.train()
        epoch_loss, epoch_f1, epoch_jac, epoch_dice = 0.0, 0.0, 0.0, 0.0
        for images, targets in tqdm(train_loader, desc="epoch {} train progress".format(epoch + 1)):
            images = images.to(device=device)
            targets = targets.to(device=device)
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_f1 += binary_f1_score(outputs, targets).item()
            epoch_jac += binary_jaccard_index(outputs, targets, threshold=0.5).item()
            epoch_dice += dice(outputs, targets.long(), threshold=0.5).item()
            del images, targets, outputs

        losses_train.append(epoch_loss / len(train_loader))
        f1_train.append(epoch_f1 / len(train_loader))
        jaccard_train.append(epoch_jac / len(train_loader))
        dice_train.append(epoch_dice / len(train_loader))

        # --- validation step --- #
        model.eval()
        epoch_loss, epoch_f1, epoch_jac, epoch_dice = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="epoch {} val progress".format(epoch + 1)):
                images = images.to(device=device)
                targets = targets.to(device=device)
                outputs = model(images)
                loss = loss_fn(outputs, targets)
                epoch_loss += loss.item()
                epoch_f1 += binary_f1_score(outputs, targets).item()
                epoch_jac += binary_jaccard_index(outputs, targets, threshold=0.5).item()
                epoch_dice += dice(outputs, targets.long(), threshold=0.5).item()
                del images, targets, outputs

        scheduler.step(epoch_loss)  # using validation loss

        losses_val.append(epoch_loss / len(val_loader))
        f1_val.append(epoch_f1 / len(val_loader))
        jaccard_val.append(epoch_jac / len(val_loader))
        dice_val.append(epoch_dice / len(val_loader))

        # --- print epoch results --- #
        log_and_print("{} epoch {}/{} metrics:".format(datetime.now(), epoch + 1, n_epochs))
        log_and_print("\t[train] loss: {:.9f}, f1_score: {:.9f}, jaccard_idx: {:.9f}, dice_score: {:.9f}".format(
            losses_train[epoch], f1_train[epoch], jaccard_train[epoch], dice_train[epoch]))
        log_and_print("\t[valid] loss: {:.9f}, f1_score: {:.9f}, jaccard_idx: {:.9f}, dice_score: {:.9f}".format(
            losses_val[epoch], f1_val[epoch], jaccard_val[epoch], dice_val[epoch]))

    # --- save weights and plot metrics --- #
    log_and_print("{} saving weights and generating plots...".format(datetime.now()))
    torch.save(model.state_dict(), os.path.join(save_path, "model_{}_weights.pth".format(model_version)))
    metrics_history = [
        ("loss", losses_train, losses_val),
        ("f1_score", f1_train, f1_val),
        ("jaccard_index", jaccard_train, jaccard_val),
        ("dice_score", dice_train, dice_val),
    ]
    print_metric_plots(metrics_history, model_version, save_path)
    log_and_print("{} training complete.".format(datetime.now()))


if __name__ == '__main__':
    # hyperparameters
    model_name = 'basic_unet'
    model_version = 3
    n_epochs = 30  # num of epochs
    batch_sz = 8  # batch size
    lr = 0.001  # learning rate (1e-3 is default for Adam)
    wd = 0.00001  # weight decay (0.0 is default for Adam)
    resize_shape = (512, 512)  # same size used in U-Net paper for training
    loss_fn_name = 'binary_cross_entropy'
    optimizer_name = 'adam'
    scheduler_name = 'reduce_on_plateau'
    seed = 222333444  # custom seed
    # seed = get_random_seed()  # generate random seed
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    list_path, save_path = get_os_dependent_paths(model_version, partition='train')

    # set up logger and deterministic seed
    setup_basic_logger(os.path.join(save_path, 'training.log'))  # initialize logger
    make_deterministic(seed)  # set deterministic seed

    # print training hyperparameters
    print_hyperparams(
        model_ver=model_version, model_name=model_name, num_epochs=n_epochs, batch_size=batch_sz, learn_rate=lr,
        weigh_decay=wd, resize_shape=resize_shape, loss_fn_name=loss_fn_name, optimizer_name=optimizer_name,
        scheduler_name=scheduler_name, seed=seed, device=device
    )

    # set up dataset(s)
    x_train, y_train, x_val, y_val = get_data_from_list(list_path, split=0.2, seed=seed)
    train_ds = CustomDS(x_train, y_train, resize_shape=resize_shape)
    val_ds = CustomDS(x_val, y_val, resize_shape=resize_shape)
    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_sz, shuffle=False)

    # compile model
    model = UNet()
    model.to(device=device)

    # init model training parameters
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # run torch summary report
    summary(model, input_size=(3, resize_shape[0], resize_shape[1]))

    # train model
    train(model, loss_fn, optimizer, scheduler, train_loader, val_loader, n_epochs, device)
