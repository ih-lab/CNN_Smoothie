import numpy as np
from collections import defaultdict

['adeno', 'squam']

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
		for lable in ['adeno', 'squam']:
			if image_name.find(lable) != -1 and t_flag:
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
	for image in evaluation:
		t_label = evaluation[image][0]
		p_label = lable_dict[np.argmax(evaluation[image][1])]
		if t_label == p_label:
			c+=1
		t+=1
		#print(image, p_label, t_label)
	acc = c * 1. / t
	return c, t, acc


def blend(preds):
	c = 0
	t = 0
	pred1 = preds[0]
	evaluation1 = pred1[0]
	for image in evaluation1:
		max_p = -1
		max_s = -1
		for p in range(len(preds)):
			if np.max(preds[p][0][image][1]) > max_s:
				max_p = p
				max_s = np.max(preds[p][0][image][1])
		t_label = evaluation1[image][0]
		pred_ = preds[max_p]
		vals_ = pred_[0][image][1]
		lable_dict = pred_[2]
		ind = np.argmax(vals_)
		p_label = lable_dict[ind]
		#print(image, t_label, p_label)
		if t_label == p_label:
			c+=1
		t+=1
	acc = c * 1. / t
	return c, t, acc


#predictions from inception
file_name = 'result.txt'
lables1 = ['adeno', 'squam']
evaluation1, lables1, lable_dict1 = read_prediction(file_name, lables1)
print(acc(evaluation1, lables1, lable_dict1))

#exit()




