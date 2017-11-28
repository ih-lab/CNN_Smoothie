import matplotlib
matplotlib.use('TKAgg')
import numpy as np
import matplotlib.pyplot as plt
import sys
import utils
import plot_utils as pu


def read_files(filename):
	x = []
	y = []

	f = open(filename, 'r')
	cnt = 0
	for line in f:
		if cnt != 0:
			line_temp = line.replace(')\n', '')
			line_temp = line_temp.replace(')', '')
			line_temp = line_temp.replace('	(', '\t')
			line_temp = line_temp.replace('	(', '\t')
			line_temp = line_temp.replace(', ', '\t')
			parsed = line_temp.split('\t')
			print(parsed)
			x.append(int(parsed[0]))
			y.append(float(parsed[3]))
		cnt+=1



	return x, y

if __name__ == '__main__':

	filenames = ['AdenoSquam', 'BladderBreastLungHe', 'BladderFourBiomarkers', 'BladderFourScores', 'BreastFiveBiomarkers', 'BreastFourScores']
	pu.figure_setup()
	fig_size = pu.get_fig_size(10, 8)
	fig = plt.figure(figsize=fig_size)
	ax = fig.add_subplot(111)
	for filename in filenames:
		x,y = read_files(filename + '.txt')
		plt.plot(x, y, label=filename)

	ax.set_xlabel('Number of Steps')
	ax.set_ylabel('Accuracy')
	ax.set_axisbelow(True)
	ax.legend(loc='lower right')
	plt.grid()
	plt.tight_layout()
	pu.save_fig(fig, '../cnn_acc_steps.pdf', 'pdf')
	plt.show()