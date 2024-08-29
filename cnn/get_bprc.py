import os
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryPrecisionRecallCurve
from tqdm import tqdm

from utils.data_import_util import get_xy_data
from custom_ds import CustomDS
from unet_model import UNet


def test(model, test_loader, device):
    global model_version, save_path, pred_ex_save_path
    bprc = BinaryPrecisionRecallCurve(thresholds=1000).to(device)
    bprc.persistent(True)
    model.eval()
    print("{} starting testing...".format(datetime.now()))

    # --- validation step --- #
    with torch.no_grad():
        for image, target in tqdm(test_loader, desc="test progress"):
            image = image.to(device=device)
            target = target.to(device=device)
            output = model(image)
            bprc.update(output, target.long())
            del image, target, output

    # --- save metric outputs --- #
    print("{} saving bprc...".format(datetime.now()))
    plt.clf()
    bprc.plot(score=True)
    plt.savefig(os.path.join(save_path, 'model_{}_test_{}.png'.format(model_version, 'bprc')))
    torch.save(bprc.state_dict(), os.path.join(save_path, '{}.pth'.format('bprc')))
    print("{} testing complete.".format(datetime.now()))


if __name__ == '__main__':
    # hyperparameters
    model_version = 3
    input_shape = (512, 512)
    dataset_name = 'synth_datasets'
    # dataset_name = 'sm_rand_spots'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set up paths and directories
    save_path = os.path.join('.', 'model_{}'.format(model_version))

    # print training hyperparameters
    print('Hyperparameters:')
    print('\tmodel_ver: {}'.format(model_version))
    print('\tinput_shape: {}'.format(input_shape))
    print('\tdataset_name: {}'.format(dataset_name))
    print('\tdevice: {}'.format(device))

    # set up dataset(s)
    x_test, y_test, _, _ = get_xy_data(dataset_name, partition='test')
    test_ds = CustomDS(x_test, y_test, dataset_name, input_shape)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # compile model
    model = UNet()
    weights_file = os.path.join(save_path, 'best_weights.pth')
    model.load_state_dict(torch.load(weights_file, map_location=device, weights_only=True))
    model.to(device=device)

    # test model
    test(model, test_loader, device)
