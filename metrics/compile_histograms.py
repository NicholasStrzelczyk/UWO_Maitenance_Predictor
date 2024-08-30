import argparse
import os
from enum import Enum

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

METRIC_NAMES = ['f1_score', 'jaccard_index']


class Experiment(Enum):
	OPTIMIZERS = 1
	GROWTH = 2
	PDM = 3


def get_experiment_paths(experiment):
	if experiment == Experiment.OPTIMIZERS:
		data_paths = [
			'../past_experiments/optimizer experiment/model_1/test results',
			'../past_experiments/optimizer experiment/model_2/test results',
			'../past_experiments/optimizer experiment/model_3/test results',
		]
		legend_names = ['AdamW', 'Adam', 'SGD']
		save_path = '../past_experiments/optimizer experiment/combined graphs'
	elif experiment == Experiment.GROWTH:
		data_paths = [
			'../past_experiments/growth experiment/model_1 30_percent/test results',
			'../past_experiments/growth experiment/model_2 50_percent/test results',
			'../past_experiments/growth experiment/model_3 normal/test results',
		]
		legend_names = ['30% growth rate', '50% growth rate', 'normal growth rate']
		save_path = '../past_experiments/10_sc growth experiment/combined graphs'
	elif experiment == Experiment.PDM:
		data_paths = [
			'../past_experiments/PdM experiment/model_1 30_percent/test results',
			'../past_experiments/PdM experiment/model_2 50_percent/test results',
			'../past_experiments/PdM experiment/model_3 normal/test results',
		]
		legend_names = ['30% growth rate', '50% growth rate', 'normal growth rate']
		save_path = '../past_experiments/4_sc growth experiment/combined graphs'
	else:
		print('Invalid experiment: {}'.format(experiment.name))
		quit()

	return {
		'data_paths': data_paths,
		'legend_names': legend_names,
		'save_path': save_path,
	}


def get_experiment_paths_v2(experiment):
	if experiment == Experiment.OPTIMIZERS:
		data_paths = [
			'../past_experiments/optimizer experiment/model_1/test results',
			'../past_experiments/optimizer experiment/model_2/test results',
			'../past_experiments/optimizer experiment/model_3/test results',
		]
		legend_names = ['SGD', 'Adam', 'AdamW']
		save_path = '../past_experiments/optimizer experiment/combined graphs'
	elif experiment == Experiment.GROWTH:
		data_paths = [
			'../past_experiments/growth experiment/model_1 30_percent/test results',
			'../past_experiments/growth experiment/model_2 50_percent/test results',
			'../past_experiments/growth experiment/model_3 normal/test results',
		]
		legend_names = ['30% growth rate', '50% growth rate', 'normal growth rate']
		save_path = '../past_experiments/10_sc growth experiment/combined graphs'
	else:
		print('Invalid experiment num: {}'.format(experiment))
		quit()

	return {
		'data_paths': data_paths,
		'legend_names': legend_names,
		'save_path': save_path,
	}


def make_combined_hists(experiment, fig_size=(6.4, 4.8), legend_loc='upper right', y_ticks=None):
	exp_dict = get_experiment_paths(experiment)

	for metric_name in METRIC_NAMES:
		plt.clf()
		plt.figure(figsize=fig_size)
		metric_lists = []

		for dir_path in exp_dict['data_paths']:
			df = pd.read_csv(os.path.join(dir_path, '{}.csv'.format(metric_name)))
			metric_lists.append(df[metric_name].tolist())

		bins = np.linspace(0.0, 1.0, 5)
		legend_names = exp_dict['legend_names']
		vals, bins, bars = plt.hist(metric_lists, bins, label=legend_names)
		plt.gca().set_xticks(bins)
		if y_ticks is not None:
			plt.gca().set_yticks(y_ticks)
		plt.xlabel(metric_name.replace('_', ' '))
		plt.ylabel('number of predictions')
		# plt.bar_label(bars)  # doesn't work for some reason???
		plt.legend(loc=legend_loc)
		plt.tight_layout()
		plt.savefig(os.path.join(exp_dict['save_path'], 'combined_{}_hist.png'.format(metric_name)))
		break  # since we're not using jaccard index right now


if __name__ == '__main__':
	# parser = argparse.ArgumentParser()
	# parser.add_argument('-exp', type=int, help='experiment number')
	# args = parser.parse_args()
	# exp = args.exp

	# hyperparameters
	exp = Experiment.OPTIMIZERS
	# exp = Experiment.GROWTH_10SC
	im_size = (4.0, 2.5)
	legend_location = 'upper center'
	y_bins = np.linspace(0, 1600, 5)

	# ----- ----- ----- #
	make_combined_hists(exp, im_size, legend_location, y_bins)

