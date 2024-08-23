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
	plt.savefig(os.path.join('.', fig_save_name))


if __name__ == '__main__':
	# hyperparameters
	num_thresholds = 1000
	bprc_path_1 = '../model_1/bprc.pth'
	bprc_path_2 = '../model_2/bprc.pth'
	bprc_path_3 = '../model_3/bprc.pth'
	fig_save_name = 'combined_bprc.png'

	# ----- ----- ----- #
	make_combined_bprc_plot(bprc_path_1, bprc_path_2, bprc_path_3)


