import os

import cv2
import numpy as np
from tqdm import tqdm

from synth_data_maker.gen_scenario_1 import interpret_sc1_timeline_csv

from synth_data_maker.synth_data_gen import (
	denoise_to_binary,
	apply_vignette,
)


def create_label(csv_data, vignette_str, mask_img_path):
	mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
	dust_cloud = apply_vignette(csv_data['dust_cloud'], vignette_str)

	pt1_x, pt1_y = csv_data['pt1_xy']
	pt2_x, pt2_y = csv_data['pt2_xy']

	target_img = np.zeros((1080, 1920, 3), dtype=np.uint8)  # blank image
	target_img[pt1_y:pt2_y, pt1_x:pt2_x, :] = dust_cloud  # insert dust cloud
	target_img = cv2.bitwise_and(target_img, target_img, mask=mask_img)  # apply metal mask
	target_img = denoise_to_binary(target_img)  # denoise & convert to b/w

	return target_img


def gen_scenario_targets(
		data_dir_path,
		dust_img_path='./image_files/dust1.png',
		mask_img_path='./image_files/metal_mask_v2.png',
):
	# step 1: create paths and interpret useful csv file contents
	timeline_file_path = os.path.join(data_dir_path, 'timeline.csv')
	label_dir_path = os.path.join(data_dir_path, 'targets')
	csv_data = interpret_sc1_timeline_csv(timeline_file_path, dust_img_path)

	# step 2: create a distinct label for each change in the timeline
	dust_target_list = [np.zeros((1080, 1920, 3), dtype=np.uint8)]
	for i in range(len(csv_data['change_days'])):
		target = create_label(csv_data, csv_data['vignettes'][i], mask_img_path)
		dust_target_list.append(target)

	# step 3: generate one label for each day
	growth_num = 0
	for day in tqdm(range(1, csv_data['total_days'] + 1), desc='Generating labels'):
		growth_num += 1 if day in csv_data['change_days'] else 0
		save_path = os.path.join(label_dir_path, 'LABEL_day_{}.png'.format(day))
		cv2.imwrite(save_path, dust_target_list[growth_num].astype(np.uint8))


if __name__ == '__main__':
	# hyperparameters
	scenario_num = 1
	data_partition = 'train'  # can be 'train', 'validation', or 'test'
	data_dir = '/Users/nick_1/Bell_5G_Data/synth_datasets/{}/scenario_{}'.format(data_partition, scenario_num)

	# ----- ----- ----- #
	gen_scenario_targets(data_dir)
	# IDEA: perhaps make a for-loop to create targets for all scenarios during one run?
