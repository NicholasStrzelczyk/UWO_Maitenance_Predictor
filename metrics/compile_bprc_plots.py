import os

import torch
from matplotlib import pyplot as plt
from torchmetrics.classification import BinaryPrecisionRecallCurve


def make_combined_bprc_plot(path1, path2, path3):
	plt.clf()
	plt.figure()
	for m_path in [path1, path2, path3]:
		bprc = BinaryPrecisionRecallCurve(thresholds=num_thresholds)
		bprc.load_state_dict(torch.load(m_path, map_location='cpu'))
		bprc.plot(score=True, ax=plt.gca())
		bprc.reset()
	plt.savefig(os.path.join(save_location, fig_save_name))


if __name__ == '__main__':
	# hyperparameters
	num_thresholds = 1000
	bprc_path_1 = '../past_experiments/Aug26_randspots_50_30_etc/model_1 (randspots30)/sm_rand_spots test results/bprc.pth'
	bprc_path_2 = '../past_experiments/Aug26_randspots_50_30_etc/model_2 (randspots50)/sm_rand_spots test results/bprc.pth'
	bprc_path_3 = '../past_experiments/Aug26_randspots_50_30_etc/model_3 (normal)/sm_rand_spots test results/bprc.pth'
	save_location = '../past_experiments/Aug26_randspots_50_30_etc/10_scenario test_results'
	fig_save_name = 'combined_bprc.png'

	# ----- ----- ----- #
	make_combined_bprc_plot(bprc_path_1, bprc_path_2, bprc_path_3)


