import os

from tqdm import tqdm

if __name__ == '__main__':
	# hyperparameters
	# data_dir = "/Users/nick_1/Bell_5G_Data/1080_snapshots/train"  # mac
	data_dir = "C:\\Users\\NickS\\UWO_Summer_Research\\Bell_5G_Data\\1080_snapshots\\train"  # windows
	list_name = "list_windows.txt"
	start_date = 13
	end_date = 28

	# ----- begin generating list ----- #
	list_file = open(os.path.join(data_dir, list_name), "a")

	for day in tqdm(range(start_date, end_date + 1), desc='Generating data list'):
		date_string = "2024_06_{}".format(day)

		for f_name in os.listdir(os.path.join(data_dir, "images")):
			if date_string in f_name:
				img_path = os.path.join(data_dir, "images", f_name)
				tgt_path = os.path.join(data_dir, "targets", "LABEL_{}.png".format(date_string))
				list_file.write(img_path + " " + tgt_path + "\n")

	list_file.close()
