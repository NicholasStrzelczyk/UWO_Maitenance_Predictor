import csv
import os
from random import randint

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from synth_data_maker.synth_data_gen import (
	remove_metal_grate,
	remove_background_dust,
	get_random_dust_variables,
	apply_vignette,
	apply_synthetic_dust,
	make_dust_region,
)


def interpret_sc2_timeline_csv(timeline_file_path, num_dust_clouds, dust_img_path='./image_files/dust1.png'):
	df = pd.read_csv(timeline_file_path)
	total_days = df['day'].values[-1]

	dust_img = cv2.imread(dust_img_path, cv2.IMREAD_COLOR)

	change_days, vignettes, clouds, cloud_locations = [], [], [], []
	for i in range(num_dust_clouds):
		changes_df = df[df['changes_occurred_{}'.format(i + 1)] == 1]
		change_days.append(changes_df['day'].values.tolist())
		vignettes.append(changes_df['vignette_strength_{}'.format(i + 1)].values.tolist())

		clouds.append(make_dust_region(
			dust_img, df['dust_rows_{}'.format(i + 1)].values[0],
			df['dust_cols_{}'.format(i + 1)].values[0]
		))

		pt1_x = df['dust_x_{}'.format(i + 1)].values[0]
		pt1_y = df['dust_y_{}'.format(i + 1)].values[0]
		pt2_x = pt1_x + clouds[i].shape[1]
		pt2_y = pt1_y + clouds[i].shape[0]
		cloud_locations.append([pt1_x, pt1_y, pt2_x, pt2_y])

	return {
		'total_days': total_days,
		'change_days': change_days,
		'dust_clouds': clouds,
		'locations': cloud_locations,
		'vignettes': vignettes,
	}


def gen_scenario_2_timeline(
		data_dir_path,
		num_days=30,
		max_dust_growths=4,
):
	num_dust_clouds = 3  # hard coded for now

	curr_probs = [0, 0, 0]
	growth_nums = [0, 0, 0]
	changes = [0, 0, 0]
	vignette_strengths = [99.9, 99.9, 99.9]
	incr_thresh = [randint(7, 13), randint(7, 13), randint(7, 13)]
	dust_vars = [
		get_random_dust_variables(max_dust_growths=max_dust_growths),
		get_random_dust_variables(max_dust_growths=max_dust_growths),
		get_random_dust_variables(max_dust_growths=max_dust_growths)
	]

	timeline = []
	for day in tqdm(range(1, num_days + 1), desc='Generating scenario 2 csv file'):

		for i in range(num_dust_clouds):
			curr_probs[i] += randint(0, 3)
			if curr_probs[i] >= incr_thresh[i] and growth_nums[i] < max_dust_growths:
				vignette_strengths[i] = dust_vars[i]['vignettes'][growth_nums[i]]
				growth_nums[i] += 1
				curr_probs[i] = 0
				changes[i] = 1
			else:
				changes[i] = 0

		line = [
			day,

			dust_vars[0]['rows'], dust_vars[0]['cols'], dust_vars[0]['loc_x'], dust_vars[0]['loc_y'],
			vignette_strengths[0], dust_vars[0]['region'], growth_nums[0], curr_probs[0], changes[0],

			dust_vars[1]['rows'], dust_vars[1]['cols'], dust_vars[1]['loc_x'], dust_vars[1]['loc_y'],
			vignette_strengths[1], dust_vars[1]['region'], growth_nums[1], curr_probs[1], changes[1],

			dust_vars[2]['rows'], dust_vars[2]['cols'], dust_vars[2]['loc_x'], dust_vars[2]['loc_y'],
			vignette_strengths[2], dust_vars[2]['region'], growth_nums[2], curr_probs[2], changes[2],
		]
		timeline.append(line)

	# write all complete timeline to the csv file
	timeline_file_path = os.path.join(data_dir_path, 'timeline.csv')
	headers = [
		'day',

		'dust_rows_1', 'dust_cols_1', 'dust_x_1', 'dust_y_1',
		'vignette_strength_1', 'region_1', 'growths_1', 'prob_1', 'changes_occurred_1',

		'dust_rows_2', 'dust_cols_2', 'dust_x_2', 'dust_y_2',
		'vignette_strength_2', 'region_2', 'growths_2', 'prob_2', 'changes_occurred_2',

		'dust_rows_3', 'dust_cols_3', 'dust_x_3', 'dust_y_3',
		'vignette_strength_3', 'region_3', 'growths_3', 'prob_3', 'changes_occurred_3',
	]
	open(timeline_file_path, 'w+').close()  # overwrite/ make new blank file
	with open(timeline_file_path, 'a', encoding='UTF8', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(headers)
		writer.writerows(timeline)


def gen_scenario_2_images(
		src_dir_path,
		data_dir_path,
		dust_img_path='./image_files/dust1.png',
):
	num_dust_clouds = 3  # hard coded for now

	timeline_file_path = os.path.join(data_dir_path, 'timeline.csv')
	im_save_dir_path = os.path.join(data_dir_path, 'images')
	hour_list = [
		'6am', '7am', '8am', '9am', '10am', '11am', '12pm',
		'1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm'
	]

	# step 1: interpret useful csv file contents
	csv_data = interpret_sc2_timeline_csv(timeline_file_path, num_dust_clouds, dust_img_path)
	dust_clouds = csv_data['dust_clouds']
	growth_nums = [0, 0, 0]

	# step 2: create the images for each day
	for day in tqdm(range(1, csv_data['total_days'] + 1), desc='Generating scenario 2 images'):

		for i in range(num_dust_clouds):
			if day in csv_data['change_days'][i]:
				dust_clouds[i] = apply_vignette(
					csv_data['dust_clouds'][i],
					csv_data['vignettes'][i][growth_nums[i]])
				growth_nums[i] += 1

		for hour in hour_list:
			img_file_name = 'day_{}_{}.png'.format(day, hour)
			img_file_path = os.path.join(src_dir_path, img_file_name)

			if os.path.isfile(img_file_path):  # skip over missing data
				image = cv2.imread(img_file_path, cv2.IMREAD_COLOR)
				background = remove_metal_grate(image)
				image = remove_background_dust(image, background)  # create a cleaned grate with no dust

				for i in range(num_dust_clouds):
					pt1_x, pt1_y, pt2_x, pt2_y = csv_data['locations'][i]

					if growth_nums[i] > 0:
						image_slice = image[pt1_y:pt2_y, pt1_x:pt2_x]
						image_slice = apply_synthetic_dust(image_slice, dust_clouds[i], stricter_blend=True)
						image[pt1_y:pt2_y, pt1_x:pt2_x] = image_slice

				cv2.imwrite(os.path.join(im_save_dir_path, 'SYNTH_{}'.format(img_file_name)), image.astype(np.uint8))


if __name__ == '__main__':
	# hyperparameters
	days_in_timeline = 28
	data_partition = 'train'  # can be 'train', 'validation', or 'test'
	src_dir = '/Users/nick_1/Bell_5G_Data/synth_datasets/src_images'
	data_dir = '/Users/nick_1/Bell_5G_Data/synth_datasets/{}/scenario_2'.format(data_partition)

	# ----- ----- ----- #
	gen_scenario_2_timeline(data_dir, num_days=days_in_timeline)
	gen_scenario_2_images(src_dir, data_dir)
