import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
	# hyperparameters
	# fig_size = (4.0, 2.5)
	fig_size = (6.4, 4.8)
	# y1_bins = np.linspace(0, 300, 5)
	# y2_bins = np.linspace(0, 300, 5)
	data_path = '../past_experiments/PdM experiment/model_3 normal/test results/SMS_test_data.csv'
	save_path = '../past_experiments/PdM experiment'

	# ----- ----- ----- #
	df = pd.read_csv(data_path)
	days = df['day'].tolist()
	sc1_scores = df['avg_f1_score_sc1'].tolist()
	sc2_scores = df['avg_f1_score_sc2'].tolist()
	sc3_scores = df['avg_f1_score_sc3'].tolist()
	foul_percentages = df['percent_img_fouling'].tolist()

	plt.clf()
	fig, ax1 = plt.subplots(figsize=fig_size)
	ax2 = ax1.twinx()

	# df['avg_f1_score'].plot(kind='bar', color='red', ax=ax1, width=0.4, position=1)
	# df['percent_img_fouling'].plot(kind='bar', color='orange', ax=ax2, width=0.4, position=0)

	ax1.plot(sc1_scores, color='blue', label='f1 score')
	ax1.plot(sc2_scores, color='orange', label='f1 score')
	ax1.plot(sc3_scores, color='green', label='f1 score')
	ax2.plot(foul_percentages, color='red', label='fouling percentage')

	ax1.set_xlabel('day')
	ax1.set_ylabel('f1 score')
	ax2.set_ylabel('percent of image containing fouling', color='red')

	plt.tight_layout()
	plt.savefig(os.path.join(save_path, 'exp3_plot.png'))
