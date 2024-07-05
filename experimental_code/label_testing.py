import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure, filters, morphology
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2gray
from tqdm import tqdm
from random import randrange


def get_source_img(data_dir, img_name):
	result = imread(os.path.join(data_dir, img_name))
	result = rgb2gray(result)  # convert to grayscale
	result = img_as_float(result)  # convert channels to float scale
	return result


def get_source_img_cv2(img_path):
	result = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	result = cv2.normalize(result, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	return result


def generate_label(src_img, fs, ps, label_threshold):
	src_img = np.pad(src_img, pad_width=((ps, ps), (ps, ps)), mode='constant')  # apply padding
	new_img = np.zeros(src_img.shape)
	for y in range(src_img.shape[0] - (2 * ps)):  # sliding window y direction
		for x in range(src_img.shape[1] - (2 * ps)):  # sliding window x direction
			win_mean = np.mean(src_img[y:(y + fs), x:(x + fs)], axis=None)  # get mean of flattened array
			new_img[y + ps, x + ps] = 1.0 if win_mean > label_threshold else 0.0
	new_img = new_img[ps:new_img.shape[0] - ps, ps:new_img.shape[1] - ps]  # remove padding
	return new_img


def make_histogram(img, plot_name):
	histogram, bin_edges = np.histogram(img, bins=256, range=(0.0, 1.0))
	fig, ax = plt.subplots()
	ax.plot(bin_edges[0:-1], histogram)
	ax.set_title("Grayscale Histogram")
	ax.set_xlabel("grayscale value")
	ax.set_ylabel("pixels")
	ax.set_xlim(0, 1.0)
	plt.savefig(plot_name)
	plt.clf()


def generate_comparison_plot(src_img, label_img):
	global img_name
	f, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

	axs[0].imshow(src_img, cmap='gray')
	axs[0].set_title('Source Image')
	axs[1].imshow(label_img, cmap='gray')
	axs[1].set_title('Labelled Image')

	plt.suptitle('Comparison Plot for \"{}\"'.format(img_name), fontsize=20)
	plt.savefig("./1080_label_test.png")


def show_img(img):
	cv2.imshow('window', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
def rotate_image(img, angle):
	image_center = tuple(np.array(img.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
	return result


# https://stackoverflow.com/questions/44816682/drawing-grid-lines-across-the-image-using-opencv-python
def draw_grid(img, grid_shape=(9, 16), color=(0, 255, 0), thickness=1):
	h, w, _ = img.shape
	rows, cols = grid_shape
	dy, dx = h / rows, w / cols
	# draw vertical lines
	for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
		x = int(round(x))
		cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)
	# draw horizontal lines
	for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
		y = int(round(y))
		cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)
	return img


def remove_metal_grate(img):
	result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	t = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	result = cv2.inpaint(img, t, 3, cv2.INPAINT_TELEA)
	return result


def remove_metal_beams(img):
	# horizontal and vertical crops
	img[:, 0:120] = 0  # left-side vertical strip
	img[:, 1800:1920] = 0  # right-side vertical strip
	img[540:690, :] = 0  # horizontal middle beam
	img[220:275, :] = 0  # horizontal upper beam

	# diagonal bars
	cv2.line(img, pt1=(120, 700), pt2=(520, 1080), color=(0, 0, 0), thickness=12)  # bottom-left diagonal bar
	cv2.line(img, pt1=(1405, 1080), pt2=(1800, 740), color=(0, 0, 0), thickness=12)  # bottom-right diagonal bar
	cv2.line(img, pt1=(120, 560), pt2=(905, 0), color=(0, 0, 0), thickness=12)  # top-left diagonal bar
	cv2.line(img, pt1=(1070, 0), pt2=(1800, 525), color=(0, 0, 0), thickness=12)  # top-right diagonal bar

	# polygons for other parts (NOT DONE, not sure if this is necessary)
	# pts = np.array([[120, 520], [180, 480], [200, 540], [120, 570]], dtype=np.int32)
	# cv2.fillPoly(image, pts=[pts], color=(0, 0, 255))
	# pts = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
	# cv2.fillPoly(image, pts=[pts], color=(0, 0, 255))

	# thicker diagonal bars (crop out entire squares)
	img[480:600, 120:240] = 0
	img[480:600, 1680:1800] = 0

	return img


def denoise_to_binary(img):
	result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	result = cv2.fastNlMeansDenoising(result, None, 20, 7, 21)
	result = cv2.threshold(result, 0, 255, cv2.THRESH_OTSU)[1]
	return result


# created July 5th, 2024 (UNUSED AS OF NOW)
# input: BGR image
# output: grayscale binary image label
def make_label(img):
	# step 1: remove the metal grate
	img = remove_metal_grate(img)
	# step 2a: rotate image so that beams are more straight
	img = rotate_image(img, 1.7)
	# step 2b: remove the straighten metal beams
	img = remove_metal_beams(img)
	# step 3: denoise the image & convert to B/W
	img = denoise_to_binary(img)
	return img


if __name__ == '__main__':
	# hyperparameters
	# threshold = 0.60  # label threshold (float value)
	filter_list = [7, 3]  # sizes must be odd numbers
	data_dir = '/Users/nick_1/Bell_5G_Data/1080_snapshots/train/images'
	label_dir = '/Users/nick_1/Bell_5G_Data/1080_snapshots/train/targets'
	grate_mask_path = './masks/1080_grate_mask_v1.png'
	roi_mask_path = './masks/1080_roi_mask_v1.png'

	# create masks
	grate_mask = img_as_float(rgb2gray(imread(grate_mask_path))) < 0.01
	roi_mask = img_as_float(rgb2gray(imread(roi_mask_path))) < 0.01

	# get a sample image
	data_list = os.listdir(data_dir)
	# img_num = randrange(0, len(data_list))
	# img_name = data_list[img_num]
	# img_name = "2024_06_14_7pm_snapshot_1.png" # rainy example
	img_name = "2024_06_13_5pm_snapshot_1.png"  # normal example
	# image = get_source_img(data_dir, img_name)

	# apply grate mask and normalize contrast
	# src = np.copy(image)

	# ---------------------------------- testing stuff
	images_list = []
	for im_name in data_list:
		if "2024_06_14" in im_name:
			images_list.append(get_source_img_cv2(os.path.join(data_dir, im_name)))

	print(len(images_list))
	image = np.average(images_list, axis=0)
	src = np.copy(image)

	image[grate_mask] = 0.0

	num_passes = 3
	for i in range(num_passes):
		image = exposure.equalize_adapthist(image, kernel_size=image.shape)
		image = filters.gaussian(image, sigma=1.0)
		t = 2 * filters.threshold_otsu(image)
		image = image > t

	footprint = morphology.rectangle(3, 3)
	image = morphology.binary_opening(image, footprint)

	# ----------------------------------

	# image[grate_mask] = 0.0
	# image[roi_mask] = 0.0  # used to ONLY label the RoI
	# image = exposure.equalize_adapthist(image, kernel_size=image.shape, clip_limit=0.99)

	# ----- begin labelling ----- #
	# for i in tqdm(range(len(filter_list))):
	# 	fs = filter_list[i]  # filter size
	# 	ps = int(np.floor(fs / 2))  # padding size
	# 	image = generate_label(image, fs, ps, threshold)

	generate_comparison_plot(src, image)

