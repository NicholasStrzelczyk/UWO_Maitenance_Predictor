import cv2
import numpy as np

if __name__ == '__main__':
	pathname = '/Users/nick_1/Bell_5G_Data/synth_datasets/test/scenario_4/targets/LABEL_day_12.png'
	target = cv2.imread(pathname, cv2.IMREAD_GRAYSCALE)
	cv2.imshow('target', target)
	cv2.waitKey(0)

	target = cv2.resize(target, (512, 512))
	cv2.imshow('target', target)
	cv2.waitKey(0)

	target = np.expand_dims(target, axis=0)  # add dimension for where channels normally are
	target = target.astype(np.float32)
	target *= (1 / 255.0)

	# ensure that the target is still binary after previous operations
	target[target < 0.5] = 0
	target[target >= 0.5] = 1

	target = np.squeeze(target)

	cv2.imshow('target', target)
	cv2.waitKey(0)

	cv2.imwrite('./example.png', 255*target)
