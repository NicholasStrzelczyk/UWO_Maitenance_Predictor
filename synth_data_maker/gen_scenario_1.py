import csv
import os
from random import randint

import cv2
import numpy as np
from tqdm import tqdm

from label_maker.label_gen import (
	list_images_for_date,
	generate_labels,
)
from synth_data_maker.synth_data_gen import (
	remove_metal_grate,
	remove_background_dust,
	get_random_dust_variables,
	make_dust_region,
	apply_vignette,
	apply_synthetic_dust,
)


def gen_scenario_1_timeline(
		data_dir_path,
		num_days=30,
		dust_incr_thresh=11,
):
	current_prob = 0  # current 'probability' value (for lack of better term)
	growth_num = 0  # number of growth increases that have occurred (max=3)
	changes = 0  # tracks whether an event occurred in the image

	dust_size, dust_location, region = get_random_dust_variables()  # stays the same during all of scenario 1
	dust_rows, dust_cols = dust_size
	dust_x, dust_y = dust_location

	timeline = []
	for day in tqdm(range(1, num_days + 1), desc='Generating scenario 1 csv file'):
		current_prob += randint(0, 3)

		if current_prob >= dust_incr_thresh and growth_num < 3:
			growth_num += 1
			current_prob = 0
			changes = 1
		else:
			changes = 0

		line = [day, dust_rows, dust_cols, dust_x, dust_y, region, growth_num, current_prob, changes]
		timeline.append(line)

	# write all complete timeline to the csv file
	timeline_file_path = os.path.join(data_dir_path, 'timeline.csv')
	headers = ['day', 'dust_rows', 'dust_cols', 'dust_x', 'dust_y', 'region', 'growths', 'prob', 'changes_occurred']
	open(timeline_file_path, 'w+').close()  # overwrite/ make new blank file
	with open(timeline_file_path, 'a', encoding='UTF8', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(headers)
		writer.writerows(timeline)


def gen_scenario_1_images(
		src_dir_path,
		data_dir_path,
		dust_img_path='./dust1.png',
		vignette_strengths=(5.0, 3.0, 2.0),
):
	timeline_file_path = os.path.join(data_dir_path, 'timeline.csv')
	im_save_dir_path = os.path.join(data_dir_path, 'images')
	dust_img = cv2.imread(dust_img_path, cv2.IMREAD_COLOR)
	hour_list = [
		'6am', '7am', '8am', '9am', '10am', '11am', '12pm',
		'1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm'
	]

	# step 1: find days when growth events took place
	num_days = 0
	change_days, change_vars = [], []
	with open(timeline_file_path, 'r', newline='') as file:
		reader = csv.reader(file)
		next(reader)  # skip the column headers
		for line in reader:
			num_days += 1
			if line[-1] == 1:  # if changes occurred on this day
				change_days.append(line[0])
				change_vars.append(line[1:5])  # append the important dust parameters
	print(num_days)

	# step 2: get universal dust cloud (since it there is only 1 for scenario 1)
	dust_vars = change_vars[-1]
	curr_dust_cloud = make_dust_region(dust_img, dust_vars[0], dust_vars[1])
	dust_cloud = np.copy(curr_dust_cloud)
	dust_h, dust_w = dust_cloud.shape[:2]
	dust_pt1_x, dust_pt1_y = int(dust_vars[2]), int(dust_vars[3])
	dust_pt2_x, dust_pt2_y = (dust_pt1_x + dust_w, dust_pt1_y + dust_h)

	# step 3: create the images for each day
	dust_change = 0
	for day in tqdm(range(1, num_days + 1), desc='Generating scenario 1 images'):

		if day in change_days:
			curr_dust_cloud = apply_vignette(dust_cloud, vignette_strengths[dust_change])
			dust_change += 1

		for hour in hour_list:
			img_file_name = 'day_{}_{}.png'.format(day, hour)
			image = cv2.imread(os.path.join(src_dir_path, img_file_name), cv2.IMREAD_COLOR)
			background = remove_metal_grate(image)
			image = remove_background_dust(image, background)  # create a cleaned grate with no dust

			if dust_change > 0:
				image_slice = image[dust_pt1_y:dust_pt2_y, dust_pt1_x:dust_pt2_x]
				image_slice = apply_synthetic_dust(image_slice, curr_dust_cloud)
				image[dust_pt1_y:dust_pt2_y, dust_pt1_x:dust_pt2_x] = image_slice

			cv2.imwrite(os.path.join(im_save_dir_path, 'SYNTH_{}'.format(img_file_name)), image.astype(np.uint8))


def gen_scenario_targets(
	data_dir_path,
	mask_path='./metal_mask_v2.png',
	num_days=30,
):
	img_dir_path = os.path.join(data_dir_path, 'images')
	label_dir_path = os.path.join(data_dir_path, 'images')

	# generate data list
	data_list = []
	for day in range(1, num_days + 1):
		common_str = 'SYNTH_day_{}'.format(day)
		day_path_list = list_images_for_date(common_str, img_dir_path)
		data_list.append((common_str, day_path_list))

	# create and save the labels
	generate_labels(data_list, mask_path, data_dir, label_dir_path, gray_labels=False)


if __name__ == '__main__':
	# hyperparameters
	vignette_strength_list = (5.0, 3.0, 2.0)
	days_in_timeline = 30
	src_dir = '/Users/nick_1/Bell_5G_Data/synth_datasets/src_images'
	data_dir = '/Users/nick_1/Bell_5G_Data/synth_datasets/train/scenario_1'

	# ----- ----- ----- #
	gen_scenario_1_timeline(data_dir, num_days=days_in_timeline)

	gen_scenario_1_images(src_dir, data_dir, vignette_strengths=vignette_strength_list)

	gen_scenario_targets(data_dir, num_days=days_in_timeline)
