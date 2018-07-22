from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy
import collections
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile


Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


"""
Extract the images into a 4D uint8 numpy array [index, y, x, depth].
Args ----> f : A file object that can be passed into a gzip reader.
Returns -> data: A 4D uint8 numpy array [index, y, x, depth].
Raises --> ValueError: If the bytestream does not start with 2051.
Addition : depth means channel number.    e.g. for grayscale depth = 1, however for RGB depth=3.
           If you are confused about magic number(2051),I powerfully advise you to google it!
"""
def extract_images(f):
    print('----> Extracting image file:', f.name,'Right Now')
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %(magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
    print('----> Extracted Successfully')
    return data


"""
Convert class labels from scalars to one-hot vectors.
"""
def dense_to_one_hot(labels_dense, num_classes):
    print('       One-Hot labels')
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    print('----> Extracted Successfully')
    return labels_one_hot


"""
Extract the labels into a 1D uint8 numpy array [index].
Args ----> f: A file object that can be passed into a gzip reader.
           one_hot: Does one hot encoding for the result.
           num_classes: Number of classes for the one hot encoding.
Returns -> labels: a 1D uint8 numpy array.
Raises --> ValueError: If the bystream doesn't start with 2049.
Addition : Similarly, google the magic number(2049) once again!
"""
def extract_labels(f, one_hot=False, num_classes=10):
    print('----> Extracting labels file:', f.name,'Right Now')
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
              raise ValueError('Invalid magic number %d in MNIST label file: %s' %(magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels, num_classes)
        print('----> Extracted Successfully')
        return labels

"""
Construct a DataSet.
one_hot arg is used only if fake_data is true.  
'dtype' can be either 'uint8' to leave the input as [0, 255],or 'float32' 
                   to rescale into '[0, 1]'.
"""
class DataSet(object):
    def __init__(self,images,labels,fake_data=False,one_hot=False,dtype=dtypes.float32,reshape=True,seed=None):
        seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
        numpy.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
            if reshape:
                assert images.shape[3] == 1
                images = images.reshape(images.shape[0],images.shape[1] * images.shape[2])
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
    #Return the next `batch_size` examples from this data set.
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]
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
            return numpy.concatenate(
                (images_rest_part, images_new_part), axis=0), numpy.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def read_data_sets(data_dir,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   fake_data=False,
                   validation_size=5000,
                   seed=None):
    print('Loading Data.')
    if fake_data:
        def fake():
            return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)
        train = fake()
        validation = fake()
        test = fake()
        return Datasets(train=train, validation=validation, test=test)

    TRAIN_IMAGES = data_dir+'/train-images-idx3-ubyte.gz'
    TRAIN_LABELS = data_dir+'/train-labels-idx1-ubyte.gz'
    TEST_IMAGES = data_dir+'/t10k-images-idx3-ubyte.gz'
    TEST_LABELS = data_dir+'/t10k-labels-idx1-ubyte.gz'

    with gfile.Open(TRAIN_IMAGES, 'rb') as f:
        train_images = extract_images(f)

    with gfile.Open(TRAIN_LABELS, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)

    with gfile.Open(TEST_IMAGES, 'rb') as f:
        test_images = extract_images(f)

    with gfile.Open(TEST_LABELS, 'rb') as f:
        test_labels = extract_labels(f, one_hot=one_hot)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError('Validation size should be between 0 and {}. Received: {}.'
                    .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = DataSet(train_images, train_labels, **options)
    validation = DataSet(validation_images, validation_labels, **options)
    test = DataSet(test_images, test_labels, **options)
    
    print('Loading Completed.','\n')
    return Datasets(train=train, validation=validation, test=test)