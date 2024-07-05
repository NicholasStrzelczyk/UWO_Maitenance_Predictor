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


def denoise_to_binary(img):
	result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	result = cv2.fastNlMeansDenoising(result, None, 20, 7, 21)
	result = cv2.threshold(result, 0, 255, cv2.THRESH_OTSU)[1]
	return result


def make_dust_region(dust_img, rows, cols):
	# step 1: create dust cloud
	h, w, c = dust_img.shape
	result = np.zeros((h * rows, w * cols, c), dust_img.dtype)

	for row in range(rows):
		for col in range(cols):
			result[row*h:(row+1)*h, col*w:(col+1)*w, :] = dust_img[:, :, :]

	# step 2: create vignette mask
	h, w, _ = result.shape
	x_resultant_kernel = cv2.getGaussianKernel(w, w / 4)
	y_resultant_kernel = cv2.getGaussianKernel(h, h / 4)
	resultant_kernel = y_resultant_kernel * x_resultant_kernel.T
	mask = resultant_kernel / resultant_kernel.max()

	# step 3: apply vignette mask
	for i in range(3):
		result[:, :, i] = result[:, :, i] * mask

	return result



if __name__ == '__main__':
	img_path = "/Users/nick_1/Bell_5G_Data/1080_snapshots/train/images/2024_06_13_2pm_snapshot_1.png"
	image = cv2.imread(img_path, cv2.IMREAD_COLOR)
	src_copy = np.copy(rotate_image(image, 1.7))
	# show_img(src_copy)

	image = remove_metal_grate(image)
	image = rotate_image(image, 1.7)
	image = remove_metal_beams(image)

	# x: 960 - 1440
	# y: 360 - 540
	# image = cv2.rectangle(image, pt1=[960, 360], pt2=[1440, 540], color=(0, 0, 255), thickness=1)
	# dust_region = np.copy(image[360:540, 960:1440])  # dims (x, y) = (480, 180)
	# dust_region = draw_grid(dust_region, grid_shape=(6, 16))
	# dust_region = cv2.rectangle(dust_region, pt1=[252, 104], pt2=[278, 138], color=(0, 0, 255), thickness=1)
	# show_img(dust_region)

	dust_anchor = [[1212, 464], [1238, 498]]  # [[x1, y1], [x2, y2]]
	dust_copy = np.copy(image[dust_anchor[0][1]:dust_anchor[1][1], dust_anchor[0][0]:dust_anchor[1][0]])
	# image = cv2.rectangle(image, pt1=dust_anchor[0], pt2=dust_anchor[1], color=(0, 0, 255), thickness=1)
	# show_img(image)
	# show_img(dust_copy)

	dust_cloud = make_dust_region(dust_copy, 3, 7)
	show_img(dust_cloud)

	# --- THE FOLLOWING CODE TAKES A DUST REGION, RESIZES IT, AND PASTES OVER THE SOURCE IMG --- #
	# dust_region_1 = np.copy(image[360:540, 960:1440])
	# # show_img(dust_region_1)  # dims (x, y) = (480, 180)
	# dust_region_2 = cv2.resize(dust_region_1, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
	# # show_img(dust_region_2)  # dims (x, y) = (600, 225) for 1.25x scale
	#
	# src_dim_y, src_dim_x = image.shape[:2]
	# dust_dim_y, dust_dim_x = dust_region_1.shape[:2]
	# resize_dim_y, resize_dim_x = dust_region_2.shape[:2]  # dims (x, y) = (720, 270) for 1.5x scale
	#
	# new_y2 = 540
	# new_y1 = new_y2 - resize_dim_y
	# new_x2 = 1440 + int((resize_dim_x - dust_dim_x) / 2)
	# new_x1 = 960 - int((resize_dim_x - dust_dim_x) / 2)
	# print(new_x1, new_x2)
	#
	# image[new_y1:new_y2, new_x1:new_x2, :] = dust_region_2[:, :, :]

	# show_img(image)

