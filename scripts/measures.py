import matplotlib
matplotlib.use('TKAgg')
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn import metrics
from sklearn.metrics import fbeta_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import zero_one_loss
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss
from sklearn.metrics import hinge_loss
from sklearn.metrics import log_loss
from sklearn.metrics import *
import utils
import plot_utils as pu
from sklearn.metrics import confusion_matrix


def read_prediction(file_name, lables):
	lable_dict = dict()
	for i in range(len(lables)):
		lable_dict[i] = lables[i]
	evaluation = defaultdict(list)
	f = open(file_name, 'r')
	for line in f:
		parsed = line.replace('\n', '').split('\t')
		score = []
		for i in range(1, len(parsed)):
			score.append(float(parsed[i]))
		image_name = parsed[0]
		t_label = -1
		image_name_ind = image_name.rfind('/')
		image_name = image_name[image_name_ind+1:]
		t_flag = True

		for lable in lables:
			if image_name.find('_' + lable) != -1 and t_flag:
				t_label = lable
				t_flag = False
		if t_flag == -1:
			print (image_name)
			print('Error')
			exit()
		#if image_name in evaluation:
			#print(image_name)
		evaluation[image_name] = [t_label, score]
	#print(len(evaluation))
	#exit()
	return evaluation, lables, lable_dict

def read_prediction_list(file_name, lables):
	y_true = []
	y_pred = []
	y_scores = []
	y_score = []
	y_label_indicator = []
	nb_size = len(lables)
	lable_dict = dict()
	for i in range(len(lables)):
		lable_dict[i] = lables[i]
	evaluation = defaultdict(list)
	f = open(file_name, 'r')
	for line in f:
		parsed = line.replace('\n', '').split('\t')
		score = []
		for i in range(1, len(parsed)):
			score.append(float(parsed[i]))
		image_name = parsed[0]
		t_label = -1
		image_name_ind = image_name.rfind('/')
		image_name = image_name[image_name_ind+1:]
		t_flag = True
		cnt = 0
		for lable in lables:
			if image_name.find('_' + lable) != -1 and t_flag:
				t_label = lable
				t_flag = False
				y_true.append(cnt)
				y_score.append(score[cnt])
				y_scores.append(score)
				tmp_lable = np.zeros(nb_size)
				tmp_lable[cnt] = 1
				y_label_indicator.append([tmp_lable])
			cnt+=1
		if t_flag == -1:
			print (image_name)
			print('Error')
			exit()
		#if image_name in evaluation:
			#print(image_name)
		y_pred.append(np.argmax(score))
		evaluation[image_name] = [t_label, score]
	#print(len(evaluation))
	#exit()
	return y_true, y_pred, y_scores, y_score, y_label_indicator

def plot_auc(y_binary, y_binary_scores, file_name, n_classes):
	pu.figure_setup()
	fig_size = pu.get_fig_size(10, 8)
	fig = plt.figure(figsize=fig_size)
	ax = fig.add_subplot(111)
	precision = dict()
	recall = dict()
	average_precision = dict()
	
	for i in range(n_classes):
		precision[i], recall[i], _ = precision_recall_curve(y_binary[:, i], y_binary_scores[:, i])
		average_precision[i] = average_precision_score(y_binary[:, i], y_binary_scores[:, i])

	precision["micro"], recall["micro"], _ = precision_recall_curve(y_binary.ravel(), y_binary_scores.ravel())
	average_precision["micro"] = average_precision_score(y_binary, y_binary_scores, average="micro")	
	print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
	plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
	         where='post')
	plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,
	                 color='b')

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title(
	    'Average precision score, micro-averaged over all classes: AUC={0:0.2f}'
	    .format(average_precision["micro"]))
	pu.save_fig(fig, 'figures/auc_' + file_name + '.pdf', 'pdf')

def plot_auc_binary(y_binary, y_binary_scores, file_name, n_classes):
	pu.figure_setup()
	fig_size = pu.get_fig_size(10, 8)
	fig = plt.figure(figsize=fig_size)
	ax = fig.add_subplot(111)
	
	precision, recall, _ = precision_recall_curve(y_binary, y_binary_scores)

	plt.step(recall, precision, color='b', alpha=0.2, where='post')
	plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	average_precision = average_precision_score(y_binary, y_binary_scores)
	plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
          average_precision))
	pu.save_fig(fig, 'figures/auc_' + file_name + '.pdf', 'pdf')


def plot_roc_binary(y_binary, y_binary_scores, file_name, n_classes):
	pu.figure_setup()
	fig_size = pu.get_fig_size(10, 8)
	fig = plt.figure(figsize=fig_size)
	ax = fig.add_subplot(111)
	

	# Compute micro-average ROC curve and ROC area
	fpr, tpr, _ = roc_curve(y_binary, y_binary_scores)
	roc_auc = auc(fpr, tpr)
	lw = 2
	plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic Curve')
	plt.legend(loc="lower right")
	pu.save_fig(fig, 'figures/roc_' + file_name + '.pdf', 'pdf')


def plot_roc(y_binary, y_binary_scores, file_name, n_classes):
	pu.figure_setup()
	fig_size = pu.get_fig_size(10, 8)
	fig = plt.figure(figsize=fig_size)
	ax = fig.add_subplot(111)
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
	    fpr[i], tpr[i], _ = roc_curve(y_binary[:, i], y_binary_scores[:, i])
	    roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_binary.ravel(), y_binary_scores.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	lw = 2
	plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
	         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic Curve')
	plt.legend(loc="lower right")
	pu.save_fig(fig, 'figures/roc_' + file_name + '.pdf', 'pdf')





def acc(evaluation, lables, lable_dict):
	c = 0
	t = 0
	lables_count = dict()
	for label in lables:
		lables_count[label] = 0
	cnt = 0
	for image in evaluation:
		t_label = evaluation[image][0]
		p_label = lable_dict[np.argmax(evaluation[image][1])]
		if t_label == p_label:
			c+=1
			lables_count[t_label]+=1
		t+=1
		
	acc = 0
	if t != 0:
		acc = c * 1. / t
	#print(lables_count)
	return c, t, acc


if __name__ == '__main__':
	#lables1 = ['ck17', 'ck56', 'egfr', 'er', 'her2']
	#lables1 = ['ck14', 'gata3', 's0084', 's100p']
	#lables1 = ['score0', 'score1', 'score2', 'score3']
	#lables1 = ['ck17', 'ck56', 'egfr', 'er', 'her2']
	#lables1 = ['adeno', 'squam']
	#lables1 = ['adeno', 'bladder', 'breast']
	directory = sys.argv[1]
	fto = open('stat/' + directory+ '.txt', 'w')
	lables1 = ['score0', 'score1', 'score2', 'score3']
	labels2 = ['score0', 'score1', 'score2', 'score3']
	labels3 = ['score0', 'score1', 'score2', 'score3']

	#lables1 = ['ck14', 'gata3', 's0084', 's100p']
	#labels2 = ['s0084', 'ck14', 's100p', 'gata3']
	#labels3 = ['s0084', 'ck14', 's100p', 'gata3']

	lables1 = ['LUAD', 'LUSC']
	labels2= ['LUAD', 'LUSC']
	labels3= ['LUAD', 'LUSC']

	#lables1 = ['adeno', 'bladder', 'breast']
	#labels2 = ['adeno', 'bladder', 'breast']
	#labels3 = ['adeno', 'breast', 'bladder']

	#lables1 = ['ck17', 'ck56', 'egfr', 'er', 'her2']
	#labels2 = ['her2', 'ck56', 'egfr', 'ck17', 'er']
	#labels3 = ['her2', 'er', 'ck17', 'egfr', 'ck56']

	#lables1 = ['adeno', 'squam']
	#labels2 = ['adeno', 'squam']
	#labels3 = ['adeno', 'squam']

	#lables1 = ['luad', 'lusc']
	#labels2 = ['luad', 'lusc']
	#labels3 = ['luad', 'lusc']


	algs = ['prediction_load_inception_v1', 'prediction_load_inception_v3', 'prediction_lastlayer_inception_resnet_v2', 'inception3_lastlayer_4000', 'inception3_lastlayer_12000', 'CNN']
	n_classes = len(lables1)
	for alg in algs:
		if alg.find('inception3_lastlayer_') != -1:
			lables1 = labels2
		if alg.find('CNN') != -1:
			lables1 = labels3

		file_name = directory + '/' + alg + '.txt'
		evaluation1, lables1, lable_dict1 = read_prediction(file_name, lables1)
		print(acc(evaluation1, lables1, lable_dict1))
		y_true, y_pred, y_scores, y_score, y_label_indicator = read_prediction_list(file_name, lables1)
		y_binary = label_binarize(y_true,classes=range(len(lables1)))

		m,n = np.shape(y_scores)
		y_binary_scores = np.zeros((m, n))
		r = 0
		for row in y_scores:
			c = 0
			for val in row:
				y_binary_scores[r, c] = val
				c+=1
			r+=1
			
		y_pred_binary = label_binarize(y_pred,classes=range(len(lables1)))
		check = []
		for (x,y) in zip(y_true, y_pred):
			if x == y:
				check.append(1.)
			else:
				check.append(0)
		print(np.sum(check) / len(check))

		fto.write(alg + '\n')
		fto.write('accuracy_score:' + '\t' + str(accuracy_score(y_true, y_pred)) + '\n')
		fto.write('cohen_kappa_score:' + '\t' + str(cohen_kappa_score(y_true, y_pred)) + '\n')
		fto.write('fbeta_score-macro:' + '\t' + str(fbeta_score(y_true, y_pred, average='macro', beta=0.5)) + '\n')
		fto.write('fbeta_score-weighted:' + '\t' + str(fbeta_score(y_true, y_pred, average='weighted', beta=0.5)) + '\n')
		fto.write('fbeta_score-micro:' + '\t' + str(fbeta_score(y_true, y_pred, average='micro', beta=0.5)) + '\n')
		fto.write('precision_score-macro:' + '\t' + str(precision_score(y_true, y_pred, average='macro')) + '\n')
		fto.write('precision_score-micro:' + '\t' + str(precision_score(y_true, y_pred, average='micro')) + '\n')
		fto.write('precision_score-weighted:' + '\t' + str(precision_score(y_true, y_pred, average='weighted')) + '\n')
		fto.write('recall_score-macro:' + '\t' + str(recall_score(y_true, y_pred, average='macro')) + '\n')
		fto.write('recall_score-micro:' + '\t' + str(recall_score(y_true, y_pred, average='micro')) + '\n')
		fto.write('recall_score-weighted:' + '\t' + str(recall_score(y_true, y_pred, average='weighted')) + '\n')
		fto.write('f1_score-macro:' + '\t' + str(f1_score(y_true, y_pred, average='macro')) + '\n')
		fto.write('f1_score-micro:' + '\t' + str(f1_score(y_true, y_pred, average='micro')) + '\n')
		fto.write('f1_score-weighted:' + '\t' + str(f1_score(y_true, y_pred, average='weighted')) + '\n')
		fto.write('jaccard_similarity_score:' + '\t' + str(jaccard_similarity_score(y_true, y_pred, normalize=True)) + '\n')
		fto.write('hamming_loss:' + '\t' + str(hamming_loss(y_true, y_pred)) + '\n')
		fto.write('log_loss:' + '\t' + str(log_loss(y_true, y_scores)) + '\n')
		

		if len(lables1) > 2:
			fto.write('average_precision_score:' + '\t' + str(average_precision_score(y_binary, y_scores)) + '\n')
			fto.write('roc_auc_score:' + '\t' + str(roc_auc_score(y_binary, y_scores)) + '\n')
		else:
			fto.write('average_precision_score:' + '\t' + str(average_precision_score(y_true, y_score)) + '\n')
			fto.write('roc_auc_score:' + '\t' + str(roc_auc_score(y_true, y_score)) + '\n')
			"""y_true_c = []
			y_score_c = []
			for l in y_true:
				if l == 0:
					y_true_c.append(1)
				else:
					y_true_c.append(0)
			for val in y_score:
				y_score_c.append(1 - val)
			#fto.write('roc_auc_score:' + '\t' + str(roc_auc_score(y_true_c, y_score_c)) + '\n')"""
		fto.write('zero_one_loss:' + '\t' + str(zero_one_loss(y_true, y_pred)) + '\n')
		y_true_array = np.asarray(y_true)
		y_pred_array = np.asarray(y_pred)
		fpr, tpr, thresholds = metrics.roc_curve(y_true_array, y_pred_array, pos_label=len(lables1) - 1)
		fto.write('AUC for' '\t' + str(metrics.auc(fpr, tpr)) + '\n')
		
		orig_stdout = sys.stdout
		sys.stdout = fto
		print(classification_report(y_true, y_pred, target_names=lables1))
		conf_mat = confusion_matrix(y_true, y_pred)
		cnt_lable = 0
		avg_rec = 0
		avg_spec = 0
		avg_per = 0
		for row in conf_mat:
			tp = row[cnt_lable]
			fp = np.sum(conf_mat[:, cnt_lable]) - tp
			fn = np.sum(conf_mat[cnt_lable, :]) - tp
			tn = np.sum(conf_mat) - tp - fp - fn
			print(lables1[cnt_lable] + " tn, fp, fn, tp: ", tn, fp, fn, tp)
			if tp != 0:
				rec = tp * 1.0 / (tp + fn)
			else:
				rec = 0
			if tp != 0:
				per = tp * 1.0 / (tp + fp)
			else:
				per = 0
			spec = tn * 1.0 / (tn + fp)
			print(lables1[cnt_lable] + " recall or sensiticity: ", rec)
			print(lables1[cnt_lable] + " specificity:", spec)
			print(lables1[cnt_lable] + " precision: ", per)
			cnt_lable += 1
			avg_rec += rec
			avg_spec += spec
			avg_per += per
		print('avg recall or sensiticity: ', avg_rec / cnt_lable)
		print('avg specificity: ', avg_spec / cnt_lable)
		print('avg precision: ', avg_per / cnt_lable)
		print('confusion_matrix for: ', lables1)
		print(confusion_matrix(y_true, y_pred))
		print('--------------------------------------------')
		print('--------------------------------------------')
		sys.stdout = orig_stdout
		fto.write('\n')
		if len(lables1) > 2:
			plot_auc(y_binary, y_binary_scores, directory + '_' + alg, n_classes)
			plot_roc(y_binary, y_binary_scores, directory + '_' + alg, n_classes)
		#else:
			#plot_auc_binary(y_true, y_score, directory + '_' + alg, n_classes)
			#plot_roc_binary(y_true, y_score, directory + '_' + alg, n_classes)
		#exit()
		#
	fto.close()