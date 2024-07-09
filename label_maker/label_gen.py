import os
from datetime import date, timedelta

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def list_images_for_date(date_str):
	results = []
	for img_name in os.listdir(data_dir):
		if date_str in img_name:
			results.append(img_name)
	return results


def create_data_list(date_list):
	results = []
	for date_str in tqdm(date_list, desc='Generating lists'):
		image_list = list_images_for_date(date_str)
		results.append((date_str, image_list))
	return results


def compute_date_avg_img(images_path_list, data_dir_path):
	img_list = []  # list to contain all images for specified date
	for img_name in images_path_list:
		img_list.append(cv2.imread(os.path.join(data_dir_path, img_name), cv2.IMREAD_COLOR))
	result = np.average(img_list, axis=0)
	return result


def remove_metal_grate(img):
	result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	t = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	result = cv2.inpaint(img, t, 3, cv2.INPAINT_TELEA)
	return result


def denoise_to_binary(img):
	result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	result = cv2.fastNlMeansDenoising(result, None, 20, 7, 21)
	result = cv2.threshold(result, 0, 255, cv2.THRESH_OTSU)[1]
	return result


def create_label_for_date(averaged_img, mask_img, date_str, label_dir_path):
	# obtain background of averaged image (remove metal grate)
	result = remove_metal_grate(averaged_img)
	result = cv2.bitwise_and(result, result, mask=mask_img)
	# denoise the image & convert to B/W binary
	result = denoise_to_binary(result)
	# save and return the new label
	cv2.imwrite(os.path.join(label_dir_path, 'LABEL_{}.png'.format(date_str)), result)
	return result


def save_label(label_img, date_str, label_dir_path, as_gray=False):
	label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY) if as_gray else label_img
	save_path = os.path.join(label_dir_path, 'LABEL_{}.png'.format(date_str))
	return cv2.imwrite(save_path, label_img.astype(np.uint8))


def generate_labels(data_list, mask_img_path, data_dir_path, label_dir_path, gray_labels=False):
	mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
	for date_str, image_list in tqdm(data_list, desc='Generating labels'):
		if len(image_list) == 0:
			print("Empty list for date {}, continuing...".format(date_str))
			continue
		else:
			# get averaged result of all images for current date
			image = compute_date_avg_img(image_list, data_dir_path)
			# generate the binary class label
			image = create_label_for_date(image, mask_img, date_str, label_dir)
			# save the label image
			save_label(image, date_str, label_dir_path, as_gray=gray_labels)


if __name__ == '__main__':
	# hyperparameters
	partition = 'train'  # for train: 06-13 to 06-28, for test: 06-29 to 06-30
	start_date = date(2024, 6, 13)
	end_date = date(2024, 6, 28)
	mask_path = './metal_mask.png'
	output_is_gray = False

	# ----- produce useful variables ----- #
	data_dir = '/Users/nick_1/Bell_5G_Data/1080_snapshots/{}/images'.format(partition)
	label_dir = '/Users/nick_1/Bell_5G_Data/1080_snapshots/{}/targets'.format(partition)
	date_range_list = pd.date_range(start_date, end_date - timedelta(days=1), freq='d').to_list()

	# ----- generate lists ----- #
	label_data_list = create_data_list(date_range_list)

	# ----- generate labels ----- #
	generate_labels(label_data_list, mask_path, data_dir, label_dir, gray_labels=output_is_gray)
