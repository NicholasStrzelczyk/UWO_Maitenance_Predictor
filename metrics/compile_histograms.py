import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

METRIC_NAMES = ['f1_score', 'jaccard_index']


def make_combined_hists(path1, path2, path3):
	for metric_name in METRIC_NAMES:
		plt.clf()
		plt.figure()
		metric_lists = []

		for dir_path in [path1, path2, path3]:
			df = pd.read_csv(os.path.join(dir_path, '{}.csv'.format(metric_name)))
			metric_lists.append(df[metric_name].tolist())

		bins = np.linspace(0.0, 1.0, 5)
		legend_names = [legend_name_1, legend_name_2, legend_name_3]
		vals, bins, bars = plt.hist(metric_lists, bins, label=legend_names)
		plt.gca().set_xticks(bins)
		plt.xlabel(metric_name)
		plt.ylabel('number of predictions')
		# plt.bar_label(bars)  # doesn't work for some reason???
		plt.legend(loc='upper right')
		plt.savefig(os.path.join(save_location, 'combined_{}_hist.png'.format(metric_name)))


if __name__ == '__main__':
	# hyperparameters
	dir_1 = '../past_experiments/Aug26_randspots_50_30_etc/model_1 (randspots30)/sm_rand_spots test results'
	dir_2 = '../past_experiments/Aug26_randspots_50_30_etc/model_2 (randspots50)/sm_rand_spots test results'
	dir_3 = '../past_experiments/Aug26_randspots_50_30_etc/model_3 (normal)/sm_rand_spots test results'

	legend_name_1 = 'randspots30'
	legend_name_2 = 'randspots50'
	legend_name_3 = 'randspots_normal'

	save_location = '../past_experiments/Aug26_randspots_50_30_etc/10_scenario test_results'

	# ----- ----- ----- #
	make_combined_hists(dir_1, dir_2, dir_3)

