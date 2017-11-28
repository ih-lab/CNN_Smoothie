from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('TKAgg')
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import matplotlib.pyplot as plt
import cv2
import numpy as np
import numpy as np
import gzip
import random as rand
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
 
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
 
class DataSet(object):
 
  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]
 
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      print(images.shape[0], images.shape[1], images.shape[2], images.shape[3])
      if reshape:
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2] * images.shape[3])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
 
  @property
  def images(self):
    return self._images
 
  @property
  def labels(self):
    return self._labels
 
  @property
  def num_examples(self):
    return self._num_examples
 
  @property
  def epochs_completed(self):
    return self._epochs_completed
 
  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]
 
 
def read_image(image_path):
    print(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (20, 20)) 
    print("image shape", image.shape)
    return np.array(image)
 
def read_labeled_image_list(image_list_file):
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
      parsed = line.replace('\n', '').split('\t')
      filenames.append(parsed[0])
      print(line)
      labels.append(int(parsed[1]))
    return len(labels), filenames, labels
 
def read_dataset(image_list, lable_list):
  nb_images = len(image_list)
  images = np.zeros((nb_images, 20, 20, 3))
  labels = np.zeros((nb_images, 2))
  for i in range(nb_images):
    images[i,:,:,:] = read_image(image_list[i])
    labels[i, lable_list[i]] = 1
  return images, labels
 
 
def read_data(image_list_file):
    fake_data = False
    one_hot = False
    dtype = dtypes.float32
    reshape = True
    seed = None
    nb_images, filenames, labels = read_labeled_image_list(image_list_file)
    print(labels)
    test_f = []
    test_l = []
    for i in range(nb_images):
        test_f.append(filenames[i])
        test_l.append(labels[i])
 
    print(nb_images)
 
    print(test_l)
    test_images, test_labels = read_dataset(test_f, test_l)
    print(test_labels)
    options = dict(dtype=dtype, reshape=reshape, seed=seed)
    test = DataSet(test_images, test_labels, **options)
    return test._images, test._labels, filenames
 
test_images, test_labels, filenames = read_data('test.txt')
sess=tf.Session()   
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('my_test_model-1000.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()
 
 
 
x = graph.get_tensor_by_name("x:0")
y_ = graph.get_tensor_by_name("y_:0")
keep_prob  = graph.get_tensor_by_name("keep_prob:0")
#W_conv1 = graph.get_tensor_by_name("W_conv1:0")
 
 
y = graph.get_tensor_by_name("y:0")
accuracy = graph.get_tensor_by_name("accuracy:0")
 
outputs = sess.run(y, feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0})
lable_names = {0: 'adeno', 1 : 'squam'} #dictionary for name of classes. You can put anything you like.
cnt = 0
filename = 'result_06.txt'
fto = open(filename, 'w')
print(outputs)
for output in outputs:
  label = np.argmax(output) #find the lable with the highest probabiliy
  score = round(np.max(output),2)
  print_out = filenames[cnt]
  print(output, len(output))
  for i in range(len(output)):
    print_out += ('\t' + str(output[i]))
  print_out += '\n'
  output
  print(print_out)
  fto.write(print_out)
  cnt+=1
fto.close()