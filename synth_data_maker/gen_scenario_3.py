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


def interpret_sc3_timeline_csv(timeline_file_path, dust_img_path='./image_files/dust1.png'):
	df = pd.read_csv(timeline_file_path)
	total_days = df['day'].values[-1]

	clean_change_days = df[df['clean_changes'] == 1]['day'].values.tolist()
	growth_change_days = df[df['growth_changes'] == 1]['day'].values.tolist()
	vignettes = df[df['growth_changes'] == 1]['vignette_strength'].values.tolist()

	dust_img = cv2.imread(dust_img_path, cv2.IMREAD_COLOR)
	cloud = make_dust_region(dust_img, df['dust_rows'].values[0], df['dust_cols'].values[0])
	pt1_x = df['dust_x'].values[0]
	pt1_y = df['dust_y'].values[0]
	pt2_x = pt1_x + cloud.shape[1]
	pt2_y = pt1_y + cloud.shape[0]

	return {
		'total_days': total_days,
		'clean_change_days': clean_change_days,
		'growth_change_days': growth_change_days,
		'dust_cloud': cloud,
		'pt1_xy': (pt1_x, pt1_y),
		'pt2_xy': (pt2_x, pt2_y),
		'vignettes': vignettes,
	}


def gen_scenario_3_timeline(
		data_dir_path,
		num_days=30,
		dust_incr_thresh=11,
		max_dust_growths=4,
		clean_interval=10,
):
	current_prob = 0  # current 'probability' value (for lack of better term)
	growth_num = 0  # number of growth increases that have occurred (max=3)
	growth_changes = 0  # tracks whether an event occurred in the image
	clean_changes = 0  # tracks when cleaning occurs
	vignette_strength = 99.9  # tracks the current vignette strength applied to the dust cloud

	# these values stay the same during all of scenario 1
	dust_vars = get_random_dust_variables(max_dust_growths=max_dust_growths)

	timeline = []
	for day in tqdm(range(1, num_days + 1), desc='Generating scenario 3 csv file'):
		current_prob += randint(0, 3)

		if day % clean_interval == 0:
			vignette_strength = 99.9
			growth_num = 0
			current_prob = 0
			clean_changes = 1
		elif current_prob >= dust_incr_thresh and growth_num < max_dust_growths:
			vignette_strength = dust_vars['vignettes'][growth_num]
			growth_num += 1
			current_prob = 0
			growth_changes = 1
		else:
			growth_changes = 0
			clean_changes = 0

		line = [
			day, dust_vars['rows'], dust_vars['cols'], dust_vars['loc_x'], dust_vars['loc_y'],
			vignette_strength, dust_vars['region'], growth_num, current_prob, growth_changes, clean_changes
		]
		timeline.append(line)

	# write all complete timeline to the csv file
	timeline_file_path = os.path.join(data_dir_path, 'timeline.csv')
	headers = [
		'day', 'dust_rows', 'dust_cols', 'dust_x', 'dust_y',
		'vignette_strength', 'region', 'growths', 'prob', 'growth_changes', 'clean_changes'
	]
	open(timeline_file_path, 'w+').close()  # overwrite/ make new blank file
	with open(timeline_file_path, 'a', encoding='UTF8', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(headers)
		writer.writerows(timeline)


def gen_scenario_3_images(
		src_dir_path,
		data_dir_path,
		dust_img_path='./image_files/dust1.png',
):
	timeline_file_path = os.path.join(data_dir_path, 'timeline.csv')
	im_save_dir_path = os.path.join(data_dir_path, 'images')
	hour_list = [
		'6am', '7am', '8am', '9am', '10am', '11am', '12pm',
		'1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm'
	]

	# step 1: interpret useful csv file contents
	csv_data = interpret_sc3_timeline_csv(timeline_file_path, dust_img_path)
	dust_cloud = csv_data['dust_cloud']
	pt1_x, pt1_y = csv_data['pt1_xy']
	pt2_x, pt2_y = csv_data['pt2_xy']
	vignettes = csv_data['vignettes']

	# step 2: create the images for each day
	growth_num = 0
	for day in tqdm(range(1, csv_data['total_days'] + 1), desc='Generating scenario 3 images'):

		if day in csv_data['clean_change_days']:
			growth_num = 0
		elif day in csv_data['growth_change_days']:
			dust_cloud = apply_vignette(csv_data['dust_cloud'], vignettes[growth_num])
			growth_num += 1

		for hour in hour_list:
			img_file_name = 'day_{}_{}.png'.format(day, hour)
			img_file_path = os.path.join(src_dir_path, img_file_name)

			if os.path.isfile(img_file_path):  # skip over missing data
				image = cv2.imread(img_file_path, cv2.IMREAD_COLOR)
				background = remove_metal_grate(image)
				image = remove_background_dust(image, background)  # create a cleaned grate with no dust

				if growth_num > 0:
					image_slice = image[pt1_y:pt2_y, pt1_x:pt2_x]
					image_slice = apply_synthetic_dust(image_slice, dust_cloud)
					image[pt1_y:pt2_y, pt1_x:pt2_x] = image_slice

				cv2.imwrite(os.path.join(im_save_dir_path, 'SYNTH_{}'.format(img_file_name)), image.astype(np.uint8))


if __name__ == '__main__':
	# hyperparameters
	days_in_timeline = 28
	growth_prob_thresh = 7
	maintenance_interval = 11
	data_partition = 'train'  # can be 'train', 'validation', or 'test'
	src_dir = '/Users/nick_1/Bell_5G_Data/synth_datasets/src_images'
	data_dir = '/Users/nick_1/Bell_5G_Data/synth_datasets/{}/scenario_3'.format(data_partition)

	# ----- ----- ----- #
	gen_scenario_3_timeline(
		data_dir,
		num_days=days_in_timeline,
		dust_incr_thresh=growth_prob_thresh,
		clean_interval=maintenance_interval
	)
	gen_scenario_3_images(src_dir, data_dir)
