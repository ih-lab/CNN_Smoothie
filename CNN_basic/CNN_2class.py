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
    # cv2.IMREAD_COLOR 
    # cv2.COLOR_BGR2GRAY 
    print(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (20, 20)) 
    

    #image = cv2.resize(image, (48, 48)) 

    print("image shape", image.shape)
    #plt.imshow(image, cmap='gray')
    #plt.show()
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
  validation_size = 5000
  seed = None
  nb_images, filenames, labels = read_labeled_image_list(image_list_file)

  combined = list(zip(filenames, labels))
  rand.shuffle(combined)
  filenames[:], labels[:] = zip(*combined)

  for i in range(10):
    print (filenames[i], labels[i])

  print (nb_images)
  nb_class = set()
  for l in labels:
    nb_class.add(l)
  print (len(nb_class))

  train_f = []
  train_l = []
  for i in range(532):
    train_f.append(filenames[i])
    train_l.append(labels[i])

  validate_f = []
  validate_l = []
  for i in range(532,759):
    validate_f.append(filenames[i])
    validate_l.append(labels[i])

  test_f = []
  test_l = []

  for i in range(759,760):
    test_f.append(filenames[i])
    test_l.append(labels[i])

  print(nb_images)

  train_images, train_labels = read_dataset(train_f, train_l)

  validation_images,  validation_labels = read_dataset(validate_f, validate_l)

  test_images, test_labels = read_dataset(test_f, test_l)

  options = dict(dtype=dtype, reshape=reshape, seed=seed)
    
  train = DataSet(train_images, train_labels, **options)
  validation = DataSet(validation_images, validation_labels, **options)
  test = DataSet(test_images, test_labels, **options)
  return base.Datasets(train=train, validation=validation, test = test)


TumorImage = read_data('train.txt')



def weight_variable(shape, name = 'noname'):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name = name)

def bias_variable(shape, name = 'noname'):
  initial = tf.constant(0.1, shape = shape)
  return tf.Variable(initial, name = name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Input layer
x  = tf.placeholder(tf.float32, [None, 1200], name='x')
y_ = tf.placeholder(tf.float32, [None, 2],  name='y_')
x_image = tf.reshape(x, [-1, 20, 20, 3])

# Convolutional layer 1
W_conv1 = weight_variable([5, 5, 3, 32], 'W_conv1')
b_conv1 = bias_variable([32], 'b_conv1')

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Convolutional layer 2
W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
b_conv2 = bias_variable([64], 'b_conv2')

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# Fully connected layer 1
h_pool2_flat = tf.reshape(h_pool2, [-1, 5*5*64])




W_fc1 = weight_variable([5 * 5 * 64, 1024], 'W_fc1')
b_fc1 = bias_variable([1024], 'b_fc1')

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob  = tf.placeholder(tf.float32, name = 'keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer 2 (Output layer)
W_fc2 = weight_variable([1024, 2], 'W_fc2')
b_fc2 = bias_variable([2], 'b_fc2')



y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')




# Evaluation functions
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

print(x_image.get_shape())
print(h_pool2.get_shape())
print(h_pool2_flat.get_shape())
print(y.get_shape())
print(h_fc1.get_shape())
print(correct_prediction.get_shape())
# Training algorithm
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Training steps

saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  max_steps = 4000
  for step in range(max_steps):
    batch_xs, batch_ys = TumorImage.train.next_batch(200)
    if (step % 50) == 0:
      saver.save(sess, 'my_test_model',global_step=1000)
      print(step, sess.run(accuracy, feed_dict={x: TumorImage.test.images, y_: TumorImage.test.labels, keep_prob: 1.0}))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
  print(max_steps, sess.run(accuracy, feed_dict={x: TumorImage.test.images, y_: TumorImage.test.labels, keep_prob: 1.0}))

  saver.save(sess, 'my_test_model', global_step=1000)

