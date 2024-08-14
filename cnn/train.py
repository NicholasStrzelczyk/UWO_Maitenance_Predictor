import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_f1_score, binary_jaccard_index
from torchsummary import summary
from tqdm import tqdm

from custom_ds import SmRandSpotsDS
from utils.data_import_util import get_data_from_list
from unet_model import UNet
from utils.log_util import log_and_print, setup_basic_logger, print_hyperparams
from utils.misc_util import print_metric_plots, get_list_path
from utils.seed_util import get_random_seed, make_deterministic


def train(model, loss_fn, optimizer, train_loader, val_loader, n_epochs, device):
    global model_name, model_version, save_path, weights_save_path, checkpoint_interval

    losses_train, losses_val = [], []
    f1_train, f1_val = [], []
    jaccard_train, jaccard_val = [], []

    # --- iterate through all epochs --- #
    log_and_print("{} starting training...".format(datetime.now()))
    for epoch in range(n_epochs):

        # --- training step --- #
        model.train()
        epoch_loss, epoch_f1, epoch_jac = 0.0, 0.0, 0.0
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
            epoch_jac += binary_jaccard_index(outputs, targets).item()
            del images, targets, outputs

        losses_train.append(epoch_loss / len(train_loader))
        f1_train.append(epoch_f1 / len(train_loader))
        jaccard_train.append(epoch_jac / len(train_loader))

        # --- validation step --- #
        model.eval()
        epoch_loss, epoch_f1, epoch_jac = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="epoch {} val progress".format(epoch + 1)):
                images = images.to(device=device)
                targets = targets.to(device=device)
                outputs = model(images)
                epoch_loss += loss_fn(outputs, targets).item()
                epoch_f1 += binary_f1_score(outputs, targets).item()
                epoch_jac += binary_jaccard_index(outputs, targets).item()
                del images, targets, outputs

        losses_val.append(epoch_loss / len(val_loader))
        f1_val.append(epoch_f1 / len(val_loader))
        jaccard_val.append(epoch_jac / len(val_loader))

        # --- print epoch results --- #
        log_and_print("{} epoch {}/{} metrics:".format(datetime.now(), epoch + 1, n_epochs))
        log_and_print("\t[train] loss: {:.9f}, f1_score: {:.9f}, jaccard_idx: {:.9f}".format(
            losses_train[epoch], f1_train[epoch], jaccard_train[epoch]))
        log_and_print("\t[valid] loss: {:.9f}, f1_score: {:.9f}, jaccard_idx: {:.9f}".format(
            losses_val[epoch], f1_val[epoch], jaccard_val[epoch]))

        # --- save weights --- #
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(weights_save_path, "e{}_weights.pth".format(epoch + 1)))

    # --- plot metrics --- #
    log_and_print("{} generating plots...".format(datetime.now()))
    metrics_history = [
        ("loss", losses_train, losses_val),
        ("f1_score", f1_train, f1_val),
        ("jaccard_index", jaccard_train, jaccard_val),
    ]
    print_metric_plots(metrics_history, model_version, save_path)
    log_and_print("{} training complete.".format(datetime.now()))


if __name__ == '__main__':
    # hyperparameters
    model_name = 'basic_unet'
    model_version = 1
    n_epochs = 100  # num of epochs
    batch_sz = 8  # batch size
    checkpoint_interval = 10  # num of epochs between save checkpoints
    input_shape = (512, 512)  # same size used in U-Net paper for training
    loss_fn_name = 'binary_cross_entropy'
    optimizer_name = 'default_adam_w'
    seed = get_random_seed()  # generate random seed
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set up paths and directories
    list_path = get_list_path(partition='train', ds_parent_folder='sm_rand_spots')
    save_path = os.path.join('.', '..', 'model_{}'.format(model_version))
    weights_save_path = os.path.join(save_path, 'weights')
    os.makedirs(weights_save_path, exist_ok=True)

    # set up logger and deterministic seed
    setup_basic_logger(os.path.join(save_path, 'training.log'))  # initialize logger
    make_deterministic(seed)  # set deterministic seed

    # print training hyperparameters
    print_hyperparams(
        model_ver=model_version, model_name=model_name, num_epochs=n_epochs, batch_size=batch_sz,
        checkpoint_interval=checkpoint_interval, input_shape=input_shape, loss_fn_name=loss_fn_name,
        optimizer_name=optimizer_name, seed=seed, device=device
    )

    # set up dataset(s)
    x_train, y_train, x_val, y_val = get_data_from_list(list_path, split=0.1, seed=seed)
    train_ds = SmRandSpotsDS(x_train, y_train)
    val_ds = SmRandSpotsDS(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_sz, shuffle=False)

    # compile model
    model = UNet()
    model.to(device=device)

    # init model optimization parameters
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(params=model.parameters())

    # run torch summary report
    summary(model, input_size=(3, input_shape[0], input_shape[1]))

    # train model
    train(model, loss_fn, optimizer, train_loader, val_loader, n_epochs, device)
