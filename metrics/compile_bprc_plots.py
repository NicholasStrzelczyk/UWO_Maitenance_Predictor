import argparse
import os
from enum import Enum

import torch
from matplotlib import pyplot as plt
from torchmetrics.classification import BinaryPrecisionRecallCurve


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
		save_path = '../past_experiments/optimizer experiment/combined graphs'
	elif experiment == Experiment.GROWTH:
		data_paths = [
			'../past_experiments/growth experiment/model_1 30_percent/test results',
			'../past_experiments/growth experiment/model_2 50_percent/test results',
			'../past_experiments/growth experiment/model_3 normal/test results',
		]
		save_path = '../past_experiments/growth experiment/combined graphs'
	elif experiment == Experiment.PDM:
		data_paths = [
			'../past_experiments/PdM experiment/model_1 30_percent/test results',
			'../past_experiments/PdM experiment/model_2 50_percent/test results',
			'../past_experiments/PdM experiment/model_3 normal/test results',
		]
		save_path = '../past_experiments/PdM experiment/combined graphs'
	else:
		print('Invalid experiment: {}'.format(experiment.name))
		quit()

	return {
		'data_paths': data_paths,
		'save_path': save_path,
	}


def get_experiment_paths_v2(experiment):
	if experiment == 1:
		data_paths = [
			'../past_experiments/optimizer experiment/model_1/test results',
			'../past_experiments/optimizer experiment/model_2/test results',
			'../past_experiments/optimizer experiment/model_3/test results',
		]
		save_path = '../past_experiments/optimizer experiment/combined graphs'
	elif experiment == 2:
		data_paths = [
			'../past_experiments/growth experiment/model_1 30_percent/test results',
			'../past_experiments/growth experiment/model_2 50_percent/test results',
			'../past_experiments/growth experiment/model_3 normal/test results',
		]
		save_path = '../past_experiments/growth experiment/combined graphs'
	else:
		print('Invalid experiment num: {}'.format(experiment))
		quit()

	return {
		'data_paths': data_paths,
		'save_path': save_path,
	}


def make_combined_bprc_plot(experiment, fig_size=(6.4, 4.8), num_thresholds=1000):
	exp_dict = get_experiment_paths(experiment)
	plt.clf()
	plt.figure(figsize=fig_size)
	for m_path in exp_dict['data_paths']:
		bprc = BinaryPrecisionRecallCurve(thresholds=num_thresholds)
		bprc.load_state_dict(torch.load(os.path.join(m_path, 'bprc.pth'), map_location='cpu'))
		bprc.plot(score=True, ax=plt.gca())
		bprc.reset()
	plt.title('')
	plt.tight_layout()
	plt.savefig(os.path.join(exp_dict['save_path'], 'combined_bprc.png'))


if __name__ == '__main__':
	# parser = argparse.ArgumentParser()
	# parser.add_argument('-exp', type=int, help='experiment number')
	# args = parser.parse_args()
	# exp = args.exp

	# hyperparameters
	exp = Experiment.OPTIMIZERS
	# exp = Experiment.GROWTH
	im_size = (4.0, 2.5)
	num_thresh = 1000  # metric states were saved w/ 1000 thresholds

	# ----- ----- ----- #
	make_combined_bprc_plot(exp, im_size, num_thresh)


