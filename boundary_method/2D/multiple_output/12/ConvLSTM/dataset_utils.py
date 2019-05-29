import os
import tensorflow as tf

in_steps = 14
out_steps = 12
HEIGHT = 512
WIDTH = 512
DEPTH = 1


class ConvLSTMDataSet(object):

  def __init__(self, data_dir='../data', subset='train', use_distortion=True):
    self.data_dir = data_dir
    self.subset = subset
    self.use_distortion = use_distortion

  def get_filenames(self):
    if self.subset in ['train', 'val', 'test']:
      return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
    else:
      raise ValueError('Invalid data subset "%s"' % self.subset)

  def parser(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    features = tf.parse_single_example(
        serialized_example,
        features={
            'inputs': tf.FixedLenFeature([in_steps * DEPTH * HEIGHT * WIDTH], tf.float32),
            'labels': tf.FixedLenFeature([out_steps * DEPTH * HEIGHT * WIDTH], tf.float32),
        })

    # Reshape from [in_steps * depth * height * width] to [in_steps, height, width, depth].
    inputs = tf.reshape(features['inputs'], [in_steps, HEIGHT, WIDTH, DEPTH])
    labels = tf.reshape(features['labels'], [out_steps, HEIGHT, WIDTH, DEPTH])

    return inputs, labels


  def make_batch(self, batch_size, shuffle=False):
    """Read the images and labels from 'filenames'."""
    filenames = self.get_filenames()
    # Repeat infinitely.
    dataset = tf.data.TFRecordDataset(filenames).repeat()

    # Parse records.
    dataset = dataset.map(
        self.parser, num_parallel_calls=batch_size)

    # Potentially shuffle records.
    if self.subset == 'train' and shuffle:
      min_queue_examples = int(
          ConvLSTMDataSet.num_examples_per_epoch(self.subset) * 0.4)
      # Ensure that the capacity is sufficiently large to provide good random
      # shuffling.
      dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

    # Batch it up.
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    inputs_batch, labels_batch = iterator.get_next()

    return inputs_batch, labels_batch

  @staticmethod
  def num_examples_per_epoch(subset='train'):
    if subset == 'train':
      return 529
    elif subset == 'val':
      return 46
    elif subset == 'test':
      return 81
    else:
      raise ValueError('Invalid data subset "%s"' % subset)
