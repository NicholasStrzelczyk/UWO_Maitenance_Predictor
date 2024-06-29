import os
from datetime import datetime

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from torchsummary import summary
from tqdm import tqdm

from custom_ds import get_split_data, BellGrayDS
from custom_model import UNet


def estimate_class_weight(y_train):
	global resize_shape
	ratio_list = []
	for path in tqdm(np.unique(y_train), desc='estimating class weights'):
		tgt = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), resize_shape)
		tgt = cv2.normalize(tgt, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		tgt[tgt >= 0.5] = 1
		tgt[tgt < 0.5] = 0
		_, count = np.unique(tgt, return_counts=True)
		ratio_list.append(float(count[1] / count[0]))
	factor = np.round(np.mean(ratio_list), decimals=4)
	return 1.0 - factor


def print_metric_plots(metrics_history):
	global model_version, save_path
	for name, m_train, m_val in metrics_history:
		plt.clf()
		plt.plot(m_train)
		plt.plot(m_val)
		plt.title("Training {}".format(name))
		plt.ylabel(name)
		plt.xlabel("epoch")
		plt.legend(['train', 'val'])
		plt.savefig(os.path.join(save_path, 'model_{}_train_{}_plot.png'.format(model_version, name)))


def create_model_directory():
	global model_version
	# new_path = "./model_v{}".format(model_version)  # mac
	new_path = ".\\model_v{}".format(model_version)  # windows
	if not os.path.exists(new_path):
		os.makedirs(new_path, exist_ok=True)
	return new_path


class FocalBCELoss(nn.Module):
	def __init__(self, alpha=None, gamma=2.0):
		super(FocalBCELoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma

	def forward(self, inputs, targets):
		bce_loss = nn.functional.binary_cross_entropy(inputs, targets)
		pt = targets * inputs + (1 - targets) * (1 - inputs)
		focal_bce = ((1.0 - pt) ** self.gamma) * bce_loss
		if self.alpha is not None:
			weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
			focal_bce = weight * focal_bce
		return focal_bce.mean()


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
	print_metric_plots(metrics_history)


# TODO:
#   - (DONE?) program more metrics to keep track of
#   - (DONE?) figure out how to add weights to BCE loss
#       --> figure out how to improve this loss function (it might not be correct)
#   - implement logging
#   - finsh the test.py file
#   - try using RGB images
#   - try classifying 3 classes
if __name__ == '__main__':
	# hyperparameters
	model_version = 2
	n_epochs = 10  # num of epochs
	batch_sz = 2  # batch size (2 works best on gpu)
	lr = 0.0001  # learning rate
	wd = 0.00001  # weight decay
	resize_shape = (512, 512)
	# data_dir = "/Users/nick_1/Bell_5G_Data/1080_snapshots"  # mac
	# list_path = os.path.join(data_dir, "train/list.txt")  # mac
	data_dir = "C:\\Users\\NickS\\UWO_Summer_Research\\Bell_5G_Data\\1080_snapshots"  # windows
	list_path = os.path.join(data_dir, "train\\list_windows.txt")  # windows
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	save_path = create_model_directory()

	# set up dataset(s)
	x_train, y_train, x_val, y_val = get_split_data(list_path, split=0.2)
	train_ds = BellGrayDS(x_train, y_train, resize_shape=resize_shape)
	val_ds = BellGrayDS(x_val, y_val, resize_shape=resize_shape)
	train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=batch_sz, shuffle=False)

	# compile model
	model = UNet(n_channels=1, n_classes=1)
	model.to(device=device)

	# init model training parameters
	class_weight_alpha = estimate_class_weight(y_train)
	print("Class weight alpha: {}".format(class_weight_alpha))
	loss_fn = FocalBCELoss(alpha=class_weight_alpha, gamma=2.0)
	optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

	# run torch summary report
	summary(model, input_size=(1, 512, 512))

	# train model
	train(model, loss_fn, optimizer, scheduler, train_loader, val_loader, n_epochs, device)
