import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure, filters, morphology
from tqdm import tqdm


def list_paths_for_date(date_string):
	path_list = []
	for img_name in os.listdir(data_dir):
		if date_string in img_name:
			path_list.append(img_name)
	return path_list


def get_source_img(img_path):
	global data_dir
	result = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	result = cv2.normalize(result, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	return result


def compute_date_avg_img(path_list):
	global data_dir
	img_list = []  # list to contain all images for specified date
	for img_name in path_list:
		img_list.append(get_source_img(os.path.join(data_dir, img_name)))
	result = np.average(img_list, axis=0)  # compute weighted average grayscale image
	return result


def compute_label_mask(img, gauss_sigma, fp, num_iters=3):
	result = img
	for i in range(num_iters):
		result = exposure.equalize_adapthist(result, kernel_size=img.shape)  # normalize contrast
		result = filters.gaussian(result, sigma=gauss_sigma)  # apply gaussian blur
		t = 2 * filters.threshold_otsu(result)  # compute OTSU threshold and double it
		result = result > t  # retain only pixels above threshold t
	result = morphology.binary_opening(result, fp)
	return result


def save_label(img, img_label_date):
	global label_dir
	save_path = os.path.join(label_dir, "LABEL_{}.png".format(img_label_date))
	plt.imsave(save_path, img, cmap='gray')


if __name__ == '__main__':

	# hyperparameters
	num_masking_passes = 3  # number of times masking operation is applied
	gaussian_sigma = 1.0  # determine amount of gaussian blur to apply
	partition = 'test'
	start_date = 29
	end_date = 30
	data_dir = '/Users/nick_1/Bell_5G_Data/1080_snapshots/{}/images'.format(partition)
	label_dir = '/Users/nick_1/Bell_5G_Data/1080_snapshots/{}/targets'.format(partition)
	grate_mask_path = './masks/1080_grate_mask_v1.png'
	footprint = morphology.rectangle(3, 3)

	# create masks
	grate_mask = get_source_img(grate_mask_path) < 0.01

	# ----- generate lists ----- #
	date_list = []
	for day in tqdm(range(start_date, end_date + 1), desc='Generating lists'):
		date_string = "2024_06_{}".format(day)
		path_list = list_paths_for_date(date_string)
		date_list.append((date_string, path_list))

	# ----- generate labels ----- #
	for date_string, path_list in tqdm(date_list, desc='Generating masks'):
		if len(path_list) == 0:
			print("Empty list for date {}".format(date_string))
			continue
		else:
			image = compute_date_avg_img(path_list)  # get averaged result of all images for current date
			image[grate_mask] = 0.0  # remove shiny metal parts
			image = compute_label_mask(image, gaussian_sigma, footprint, num_masking_passes)  # generate the binary class label
			save_label(image, date_string)  # save the label image

