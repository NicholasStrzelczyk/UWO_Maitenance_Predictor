import os
import shutil
from datetime import date, timedelta

import pandas as pd
from tqdm import tqdm


def exists_in_dir(f_str, dir_path):
	exists = False
	for f_name in os.listdir(dir_path):
		if f_str in f_name:
			exists = True
	return exists


def copy_paste_buffer(buf):
	for src, dst in buf:
		shutil.copyfile(src, dst)


if __name__ == '__main__':
	# hyperparameters
	start_date = date(2024, 6, 13)  # range: 06-13 to 07-14 (inclusively)
	end_date = date(2024, 7, 15)  # end date is not inclusive
	src_dir = '/Users/nick_1/Bell_5G_Data/all_1080_data/'
	dest_dir = '/Users/nick_1/Bell_5G_Data/synth_datasets/src_images'

	# time and date lists
	hour_list = [
		'6am', '7am', '8am', '9am', '10am', '11am', '12pm',
		'1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm'
	]
	date_range_list = pd.date_range(start_date, end_date - timedelta(days=1), freq='d').to_list()

	# --- obtain path for image and copy/ rename it --- #
	day_num = 1
	dates_not_found = []
	files_not_found = []

	for timestamp in tqdm(date_range_list, desc='Copying images'):
		date_str = str(timestamp).split(' ')[0].replace('-', '_').strip()
		files_found_for_date = 0
		file_copy_buffer = []

		for hour in hour_list:
			file_name = '{}_{}_snapshot_1.png'.format(date_str, hour)
			file_path = os.path.join(src_dir, file_name)
			dest_path = os.path.join(dest_dir, 'day_{}_{}.png'.format(day_num, hour))

			if exists_in_dir(file_name, src_dir):
				file_copy_buffer.append((file_path, dest_path))
				files_found_for_date += 1
			else:
				files_not_found.append(file_name)

		if files_found_for_date > 0:
			copy_paste_buffer(file_copy_buffer)
			day_num += 1
		else:
			dates_not_found.append(date_str)
	# ------------------------------------------------- #

	print('Could not find these specific files:')
	for file in files_not_found:
		print('\t{}'.format(file))

	print('0 files were found for these dates:')
	for date_str in dates_not_found:
		print('\t{}'.format(date_str))

