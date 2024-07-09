import math
import os
from random import randint

import cv2
import numpy as np
from tqdm import tqdm


def get_random_dust_variables(
		dust_img_path='./dust1.png',
		max_cols=35,
		max_rows=10
):
	rows = randint(3, max_rows)
	cols = randint(rows, max_cols)

	dust_h, dust_w = cv2.imread(dust_img_path).shape[:2]
	max_loc_y = 1080 - (rows * dust_h) - 100
	max_loc_x = 1920 - (cols * dust_w) - 100

	location_y = randint(100, max_loc_y)
	location_x = randint(100, max_loc_x)

	return (rows, cols), (location_x, location_y)


def show_img(img):
	cv2.imshow('window', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def remove_metal_grate(img):
	result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	t = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	result = cv2.inpaint(img, t, 3, cv2.INPAINT_TELEA)
	return result


def remove_background_dust(img, background):
	result = img - background
	return result


def make_dust_region(dust_img, rows, cols, vignette_strength):
	# step 1: create dust cloud
	h, w, c = dust_img.shape
	result = np.zeros((h * rows, w * cols, c), dust_img.dtype)

	for row in range(rows):
		for col in range(cols):
			result[row*h:(row+1)*h, col*w:(col+1)*w, :] = dust_img[:, :, :]

	# step 2: create vignette mask
	h, w, _ = result.shape
	x_resultant_kernel = cv2.getGaussianKernel(w, w / vignette_strength)
	y_resultant_kernel = cv2.getGaussianKernel(h, h / vignette_strength)
	resultant_kernel = y_resultant_kernel * x_resultant_kernel.T
	mask = resultant_kernel / resultant_kernel.max()

	# step 3: apply vignette mask
	for i in range(3):
		result[:, :, i] = result[:, :, i] * mask

	return result


def apply_synthetic_dust(img1, img2, alpha=0.99, stricter_blend=False):
	gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	thresh_img1 = cv2.threshold(gray_img1, 0, 255, cv2.THRESH_OTSU)[0]  # OTSU for img1
	thresh_img2 = cv2.threshold(gray_img2, 0, 255, cv2.THRESH_OTSU)[0]  # OTSU for img2
	result = np.zeros(img1.shape, np.uint8)
	for y in range(result.shape[0]):  # loop through pixels in y-axis
		for x in range(result.shape[1]):  # loop through pixels in x-axis
			for c in range(result.shape[2]):  # loop through color channels
				if gray_img1[y, x] < thresh_img1:  # if this pixel belongs to the background of img1 (not metal grate)
					if stricter_blend:  # this preserves more of the original image's background noise
						if gray_img2[y, x] >= thresh_img2:  # blend pixels according to alpha
							result[y, x, c] = ((1 - alpha) * img1[y, x, c]) + (alpha * img2[y, x, c])
						else:  # makes pixel blend more fair toward edges of vignette
							result[y, x, c] = (0.5 * img1[y, x, c]) + (0.5 * img2[y, x, c])
					else:  # blend pixels according to alpha
						result[y, x, c] = ((1 - alpha) * img1[y, x, c]) + (alpha * img2[y, x, c])
				else:  # else this pixel belongs to the foreground of img1 (the metal grate)
					result[y, x, c] = img1[y, x, c]  # color pixel same as img1
	return result


def create_synthetic_data(
		src_path,
		metal_mask_path='./metal_mask.png',
		dust_img_path='./dust1.png',
		dust_vignette_strength=3.0,
		dust_cloud_size=(7, 17),
		dust_location=(600, 270),
		blend_alpha=0.99
):
	# step 1: read images
	src_img = cv2.imread(src_path, cv2.IMREAD_COLOR)
	mask_img = cv2.imread(metal_mask_path, cv2.IMREAD_GRAYSCALE)
	dust_img = cv2.imread(dust_img_path, cv2.IMREAD_COLOR)

	# step 2: preprocess source image
	background = remove_metal_grate(src_img)
	background = cv2.bitwise_and(background, background, mask=mask_img)
	result = remove_background_dust(src_img, background)

	# step 3: generate dust cloud from sample
	dust_patten_img = make_dust_region(dust_img, dust_cloud_size[0], dust_cloud_size[1], dust_vignette_strength)
	h, w = dust_patten_img.shape[:2]
	pt1_x, pt1_y = dust_location
	pt2_x, pt2_y = (dust_location[0] + w, dust_location[1] + h)

	# step 4: apply dust cloud to the background of the image
	img_slice = result[pt1_y:pt2_y, pt1_x:pt2_x]
	img_slice = apply_synthetic_dust(img_slice, dust_patten_img, blend_alpha)
	result[pt1_y:pt2_y, pt1_x:pt2_x] = img_slice

	return result


if __name__ == '__main__':
	# hyperparameters
	image_limit = False
	max_images = 10  # only used if 'image_limit' is True
	vignette_strength_list = [5.0, 3.0, 2.0]
	data_dir = '/Users/nick_1/Bell_5G_Data/1080_snapshots/train/images'
	save_dir = '/Users/nick_1/Bell_5G_Data/1080_snapshots/train/synth_images'
	# mask_path = './metal_mask.png'
	# dust_path = './dust1.png'

	# ---------------------------------------- #
	img_count = 0
	for img_name in tqdm(os.listdir(data_dir), desc='Generating synthetic images'):
		# obtain randomly generated synthetic data parameters
		size, location = get_random_dust_variables()

		# create and save synthetic data objects for varying vignette strengths
		for i in range(len(vignette_strength_list)):
			synth_img = create_synthetic_data(
				os.path.join(data_dir, img_name),
				dust_vignette_strength=vignette_strength_list[i],
				dust_cloud_size=size,
				dust_location=location,
			)
			cv2.imwrite(os.path.join(data_dir, 'SYNTH{}_{}'.format(i+1, img_name)), synth_img.astype(np.uint8))
		img_count += 1

		if image_limit and img_count >= max_images:
			break

	# ---------------------------------------- #
	# img_name = '2024_06_13_2pm_snapshot_1.png'
	# img_path = os.path.join(data_dir, img_name)
	#
	# vignette_strength_list = [5.0, 3.0, 2.0]
	# for vs in vignette_strength_list:
	# 	test = create_synthetic_data(img_path, dust_vignette_strength=vs, dust_cloud_size=(7, 33))
	# 	show_img(test)
