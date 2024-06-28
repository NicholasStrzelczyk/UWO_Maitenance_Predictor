import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def get_split_data(list_path, split=0.2):
	all_data = []
	for line in open(list_path, "r"):
		all_data.append(line)
	train_data, test_data = train_test_split(all_data, test_size=split, random_state=42, shuffle=True)

	x_train, y_train = [], []
	for d in train_data:
		x, y = d.split(" ")
		x_train.append(x.strip()), y_train.append(y.strip())

	x_test, y_test = [], []
	for d in test_data:
		x, y = d.split(" ")
		x_test.append(x.strip()), y_test.append(y.strip())

	return x_train, y_train, x_test, y_test


# TODO:
#   - make an self.augment() function using cv2 transforms
#   - decide whether to use RGB or grayscale
class BellGrayDS(Dataset):
	def __init__(self, x_set, y_set, resize_shape=(512, 512), as_gray=True, img_as_float=True, img_transpose=None):
		self.img_transpose = img_transpose  # (2, 0, 1) for rgb
		self.img_as_float = img_as_float
		self.as_gray = as_gray
		self.resize_shape = resize_shape
		self.x = x_set
		self.y = y_set

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		image = cv2.imread(self.x[idx], cv2.IMREAD_GRAYSCALE)
		target = cv2.imread(self.y[idx], cv2.IMREAD_GRAYSCALE)
		image, target = self.preprocess(image, target)
		return image, target

	def preprocess(self, img, tgt):
		if self.resize_shape is not None:
			img = cv2.resize(img, self.resize_shape)
			tgt = cv2.resize(tgt, self.resize_shape)
		if self.img_as_float:
			img = cv2.normalize(img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
			tgt = cv2.normalize(tgt, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		if self.img_transpose is not None:
			img = np.transpose(img, axes=self.img_transpose)
			tgt = np.transpose(tgt, axes=self.img_transpose)
		if self.as_gray:
			img = np.expand_dims(img, axis=0)
			tgt = np.expand_dims(tgt, axis=0)

		tgt[tgt >= 0.5] = 1
		tgt[tgt < 0.5] = 0

		return img, tgt
