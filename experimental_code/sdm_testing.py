import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import filters
from skimage import exposure


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


def create_metal_beam_mask(out_path='./metal_mask.png'):
	result = np.ones(shape=(1080, 1920), dtype=np.uint8)
	result *= 255
	result = remove_metal_beams(result)
	result = rotate_image(result, 358.3)
	cv2.imwrite(out_path, result)
	return result


def denoise_to_binary(img):
	result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	result = cv2.fastNlMeansDenoising(result, None, 20, 7, 21)
	result = cv2.threshold(result, 0, 255, cv2.THRESH_OTSU)[1]
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


def remove_background_dust(img, background):
	result = img - background
	return result


def create_new_dust_img(src, out_path='./dust_img.png', coords=((1215, 465), (1235, 501)), contrast_alpha=1.3):
	# x: 960 - 1440
	# y: 360 - 540
	# image = cv2.rectangle(image, pt1=[960, 360], pt2=[1440, 540], color=(0, 0, 255), thickness=1)
	# dust_region = np.copy(image[360:540, 960:1440])  # dims (x, y) = (480, 180)
	# dust_region = draw_grid(dust_region, grid_shape=(6, 16))
	# dust_region = cv2.rectangle(dust_region, pt1=[252, 104], pt2=[278, 138], color=(0, 0, 255), thickness=1)
	# show_img(dust_region)

	pt1, pt2 = coords  # coords = ((x1, y1), (x2, y2))
	pt1_x, pt1_y = pt1  # pt1 represents the upper-left corner coordinates (x, y)
	pt2_x, pt2_y = pt2  # pt2 represents the bottom-right corner coordinates (x, y)
	result = np.copy(src[pt1_y:pt2_y, pt1_x:pt2_x])
	result = cv2.convertScaleAbs(result, alpha=contrast_alpha, beta=0)  # contrast control alpha (1.0 - 3.0)
	cv2.imwrite(out_path, result.astype(np.uint8))
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
	data_dir = '/Users/nick_1/Bell_5G_Data/1080_snapshots/train/images'
	img_name = '2024_06_13_2pm_snapshot_1.png'
	img_path = os.path.join(data_dir, img_name)

	vignette_strength_list = [5.0, 3.0, 2.0]
	for vs in vignette_strength_list:
		test = create_synthetic_data(img_path, dust_vignette_strength=vs, dust_cloud_size=(7, 33))
		show_img(test)
