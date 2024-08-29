import os
from enum import Enum

import torch
from matplotlib import pyplot as plt
from torchmetrics.classification import BinaryPrecisionRecallCurve


class Experiment(Enum):
	OPTIMIZERS = 1
	GROWTH_10SC = 2
	GROWTH_4SC = 3


def get_experiment_paths(experiment):
	if experiment == Experiment.OPTIMIZERS:
		data_paths = [
			'../past_experiments/optimizer experiment/model_1/test results',
			'../past_experiments/optimizer experiment/model_2/test results',
			'../past_experiments/optimizer experiment/model_3/test results',
		]
		save_path = '../past_experiments/optimizer experiment/combined graphs'
	elif experiment == Experiment.GROWTH_10SC:
		data_paths = [
			'../past_experiments/10_sc growth experiment/model_1 30_percent/test results',
			'../past_experiments/10_sc growth experiment/model_2 50_percent/test results',
			'../past_experiments/10_sc growth experiment/model_3 normal/test results',
		]
		save_path = '../past_experiments/10_sc growth experiment/combined graphs'
	elif experiment == Experiment.GROWTH_4SC:
		data_paths = [
			'../past_experiments/4_sc growth experiment/model_1 30_percent/test results',
			'../past_experiments/4_sc growth experiment/model_2 50_percent/test results',
			'../past_experiments/4_sc growth experiment/model_3 normal/test results',
		]
		save_path = '../past_experiments/4_sc growth experiment/combined graphs'
	else:
		print('Invalid experiment: {}'.format(experiment.name))
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
	# hyperparameters
	# exp = Experiment.OPTIMIZERS
	exp = Experiment.GROWTH_10SC
	# exp = Experiment.GROWTH_4SC
	im_size = (4.0, 2.5)
	num_thresh = 1000  # metric states were saved w/ 1000 thresholds

	# ----- ----- ----- #
	make_combined_bprc_plot(exp, im_size, num_thresh)


