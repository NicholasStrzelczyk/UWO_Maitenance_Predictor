import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

METRIC_NAMES = ['f1_score', 'jaccard_index']


def make_combined_hists(path1, path2, path3):
	for metric_name in METRIC_NAMES:
		plt.clf()
		plt.figure()
		plt.style.use('seaborn-deep')
		metric_lists = []

		for dir_path in [path1, path2, path3]:
			df = pd.read_csv(os.path.join(dir_path, '{}.csv'.format(metric_name)))
			metric_lists.append(df[metric_name].tolist())

		bins = np.linspace(0.0, 1.0, 4)
		legend_names = [legend_name_1, legend_name_2, legend_name_3]
		_, _, bars = plt.hist(metric_lists, bins, label=legend_names)
		plt.bar_label(bars)
		plt.xlabel(metric_name)
		plt.ylabel('number of predictions')
		plt.legend(loc='upper right')
		plt.savefig(os.path.join('.', 'combined_{}_hist.png'.format(metric_name)))


if __name__ == '__main__':
	# hyperparameters
	dir_1 = '../model_1'
	dir_2 = '../model_2'
	dir_3 = '../model_3'

	legend_name_1 = 'AdamW'
	legend_name_2 = 'Adam'
	legend_name_3 = 'SGD'

	# ----- ----- ----- #
	make_combined_hists(dir_1, dir_2, dir_3)

