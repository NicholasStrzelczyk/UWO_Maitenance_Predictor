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
	region_num = randint(1, 2)  # determines whether dust will be in top or bottom region (1 or 2)

	# valid region 1 = [x: 120-1840, y: 0-510]
	# valid region 2 = [x: 120-1840, y: 700-1080]
	dust_h, dust_w = cv2.imread(dust_img_path).shape[:2]  # [width=20, height=36]
	min_location_y = 0 if region_num == 1 else 700
	max_location_y = 510 - (rows * dust_h) if region_num == 1 else 1080
	min_location_x = 120
	max_location_x = 1840 - (cols * dust_w)

	location_y = randint(min_location_y, max_location_y)
	location_x = randint(min_location_x, max_location_x)

	return (rows, cols), (location_x, location_y), region_num


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


def make_dust_region(dust_img, rows, cols):
	h, w, c = dust_img.shape
	result = np.zeros((h * rows, w * cols, c), dust_img.dtype)
	for row in range(rows):
		for col in range(cols):
			result[row * h:(row + 1) * h, col * w:(col + 1) * w, :] = dust_img[:, :, :]
	return result


def apply_vignette(img, strength):
	result = np.copy(img)
	h, w = result.shape[:2]
	x_resultant_kernel = cv2.getGaussianKernel(w, w / strength)
	y_resultant_kernel = cv2.getGaussianKernel(h, h / strength)
	resultant_kernel = y_resultant_kernel * x_resultant_kernel.T
	mask = resultant_kernel / resultant_kernel.max()
	for i in range(3):
		result[:, :, i] = result[:, :, i] * mask
	return result


def apply_synthetic_dust(img1, img2, alpha=0.99):
	gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	# gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	thresh_img1 = cv2.threshold(gray_img1, 0, 255, cv2.THRESH_OTSU)[0]  # OTSU for img1
	# thresh_img2 = cv2.threshold(gray_img2, 0, 255, cv2.THRESH_OTSU)[0]  # OTSU for img2
	result = np.zeros(img1.shape, np.uint8)
	for y in range(result.shape[0]):  # loop through pixels in y-axis
		for x in range(result.shape[1]):  # loop through pixels in x-axis
			for c in range(result.shape[2]):  # loop through color channels
				if gray_img1[y, x] < thresh_img1:  # if this pixel belongs to the background of img1 (not metal grate)
					# if stricter_blend:  # this preserves more of the original image's background noise
					# 	if gray_img2[y, x] >= thresh_img2:  # blend pixels according to alpha
					# 		result[y, x, c] = ((1 - alpha) * img1[y, x, c]) + (alpha * img2[y, x, c])
					# 	else:  # makes pixel blend more fair toward edges of vignette
					# 		result[y, x, c] = (0.5 * img1[y, x, c]) + (0.5 * img2[y, x, c])
					# else:  # blend pixels according to alpha
					result[y, x, c] = ((1 - alpha) * img1[y, x, c]) + (alpha * img2[y, x, c])
				else:  # else this pixel belongs to the foreground of img1 (the metal grate)
					result[y, x, c] = img1[y, x, c]  # color pixel same as img1
	return result


def create_synthetic_data(
		src_path,
		metal_mask_path='./metal_mask_v2.png',
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
	dust_patten_img = make_dust_region(dust_img, dust_cloud_size[0], dust_cloud_size[1])
	dust_patten_img = apply_vignette(dust_patten_img, dust_vignette_strength)
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
	vignette_strength_list = [5.0, 3.0, 2.0]
	data_dir = '/Users/nick_1/Bell_5G_Data/1080_snapshots/train/images'
	save_dir = '/Users/nick_1/Bell_5G_Data/1080_snapshots/train/synth_images'
	# mask_path = './metal_mask.png'
	# dust_path = './dust1.png'

	# ---------------------------------------- #
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
