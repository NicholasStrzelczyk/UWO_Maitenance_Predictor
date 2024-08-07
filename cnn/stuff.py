import cv2
import numpy as np
import sklearn.metrics as m
import torch
from torchmetrics.functional.classification import binary_f1_score

if __name__ == '__main__':
	# pathname = '/Users/nick_1/Bell_5G_Data/synth_datasets/test/scenario_4/targets/LABEL_day_12.png'
	# target = cv2.imread(pathname, cv2.IMREAD_GRAYSCALE)
	# cv2.imshow('target', target)
	# cv2.waitKey(0)
	#
	# target = cv2.resize(target, (512, 512))
	# cv2.imshow('target', target)
	# cv2.waitKey(0)
	#
	# target = np.expand_dims(target, axis=0)  # add dimension for where channels normally are
	# target = target.astype(np.float32)
	# target *= (1 / 255.0)
	#
	# # ensure that the target is still binary after previous operations
	# target[target < 0.5] = 0
	# target[target >= 0.5] = 1
	#
	# target = np.squeeze(target)
	#
	# cv2.imshow('target', target)
	# cv2.waitKey(0)
	#
	# cv2.imwrite('./example.png', 255*target)

	pred_path = '/Users/nick_1/PycharmProjects/Western Summer Research/UWO_Maitenance_Predictor/model_3/pred_examples/f1/f1_pred_54.png'
	tgt_path = '/Users/nick_1/PycharmProjects/Western Summer Research/UWO_Maitenance_Predictor/model_3/pred_examples/f1/f1_targ_54.png'

	pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
	tgt = cv2.imread(tgt_path, cv2.IMREAD_GRAYSCALE)

	pred = pred.astype(np.float32)
	tgt = tgt.astype(np.float32)
	pred *= (1 / 255.0)
	tgt *= (1 / 255.0)

	print(pred)
	print(tgt)
	cv2.imshow('abc', tgt)
	cv2.waitKey(0)

	print(m.f1_score(tgt.astype(int).flatten(), pred.astype(int).flatten()))

	pred = torch.tensor(pred)
	tgt = torch.tensor(tgt)

	f1 = binary_f1_score(pred, tgt).item()
	print(f1)



