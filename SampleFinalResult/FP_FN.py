import matplotlib
matplotlib.use('TKAgg')
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import plot_utils as pu


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
		evaluation[image_name] = [t_label, score]
	return evaluation, lables, lable_dict





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
		t+=1
		
	acc = 0
	if t != 0:
		acc = c * 1. / t
	return c, t, acc

def TPR_FPR(evaluation, lables, lable_dict, thr):
	c = 0
	t = 0
	lables_count = dict()
	for label in lables:
		lables_count[label] = 0
	cnt = 0
	FP = 0
	FN = 0
	TP = 0
	TN = 0
	for image in evaluation:
		t_label = evaluation[image][0]
		scores = evaluation[image][1]
		#if np.max(evaluation[image][1]) >= thr:
		if evaluation[image][1][0] > thr:
			p_label = lable_dict[0]
			if t_label == p_label:
				TP+=1
			else:
				FP+=1
		else:
			p_label = lable_dict[1]
			if t_label == p_label:
				TN+=1
			else:
				FN+=1

	return TP, FP, TN, FN

def TPR_FPR_01(evaluation, lables, lable_dict, thr):
	c = 0
	t = 0
	lables_count = dict()
	for label in lables:
		lables_count[label] = 0
	cnt = 0
	FP = 0
	FN = 0
	TP = 0
	TN = 0
	for image in evaluation:
		t_label = evaluation[image][0]
		scores = evaluation[image][1]
		#if np.max(evaluation[image][1]) >= thr:
		if evaluation[image][1][1] > thr:
			p_label = lable_dict[1]
			if t_label == p_label:
				TP+=1
			else:
				FP+=1
		else:
			p_label = lable_dict[0]
			if t_label == p_label:
				TN+=1
			else:
				FN+=1

	return TP, FP, TN, FN


def clean_p(p):
	c_p = []
	c_flag = False
	for v in p:
		if v < 1 and not c_flag:
			c_p.append(v)
		else:
			c_p.append(1)
			c_flag = True
	return c_p

lables1 = ['adeno', 'squam']



#exit()



lable_names = ['CNN', 'Inception-ResNet-v2 (last layer)', 'Inception-v3 (last layer)', 'Inception-v3 (last layer)', 'Inception-v3 (fine-tune)', 'Inception-v1 (fine-tune)']
file_names = ['CNN.txt', 'prediction_lastlayer_inception_resnet_v2.txt', 'inception3_lastlayer_4000.txt', 'inception3_lastlayer_12000.txt', 'prediction_load_inception_v3.txt', 'prediction_load_inception_v1.txt']

bins = []
bins.append(0.0)
bins.append(0.00001)


nb_bins = 10000
for i in range(1, nb_bins):
	thr = ( 1. * i  / nb_bins)
	bins.append(thr)
print (bins)


bins.append(0.9999)
bins.append(0.99991)
bins.append(0.99992)
bins.append(0.99993)
bins.append(0.99994)
bins.append(0.99995)
bins.append(0.99996)
bins.append(0.99997)
bins.append(0.99998)
bins.append(0.99999)
bins.append(1.0)


pu.figure_setup()
fig_size = pu.get_fig_size(12, 10)
fig = plt.figure(figsize=fig_size)
ax = fig.add_subplot(111)

print(bins)
cnt = len(file_names) - 1
for file_name in reversed(file_names):
	#print (file_name)
	evaluation1, lables1, lable_dict1 = read_prediction(file_name, lables1)
	#print(acc(evaluation1, lables1, lable_dict1))

	p = []
	r = []
	FPR_0  = 0
	FPR1_0  = 0
	roc = 1
	for thr in bins:
		TP, FP, TN, FN = TPR_FPR(evaluation1, lables1, lable_dict1,thr)
		TPR = TP * 1.0 / (TP + FN)
		FPR = FP * 1.0 / (FP + TN)

		TP1, FP1, TN1, FN1 = TPR_FPR_01(evaluation1, lables1, lable_dict1,thr)
		TPR1 = TP1 * 1.0 / (TP1 + FN1)
		FPR1 = FP1 * 1.0 / (FP1 + TN1)

		p.append((TPR1 + TPR) / 2.0)
		r.append((FPR1 + FPR) / 2.0)
		roc += (0.5 * (FPR_0 - FPR) * TPR)
		roc += (0.5 * (FPR1_0 - FPR1) * TPR1)
		FPR_0 = FPR
		FPR1_0 = FPR1
	print(file_name, roc)
	#print(p)
	#print(r)
	plt.plot(r, p, label=lable_names[cnt] + ', AUC=' + str(round(roc,2)))
	cnt -= 1

#plt.title('')
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.ylim([0., 1.05])
ax.legend(loc='lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid()
plt.tight_layout()
pu.save_fig(fig, 'adenosquam_FPR_TPR.pdf', 'pdf')
plt.show()



