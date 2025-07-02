def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, str):
        value = value.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

example = tf.train.Example(
    features=tf.train.Features(
        feature={
            'height': _int64_feature(128),
            'width': _int64_feature(128),
            'depth': _int64_feature(3),
            'label': _int64_feature(7),
            'image_raw': _bytes_feature(image_bytes),
        }
    )
)

feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_function(proto):
    return tf.io.parse_single_example(proto, feature_description)

# Load TFRecord dataset
raw_dataset = tf.data.TFRecordDataset('data.tfrecord')
parsed_dataset = raw_dataset.map(_parse_function)

import tensorflow as tf
import numpy as np

image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
label = 5

# Serialize the image
image_raw = image.tobytes()

# Create example
example = tf.train.Example(features=tf.train.Features(feature={
    'image': _bytes_feature(image_raw),
    'label': _int64_feature(label),
}))

# Write to TFRecord
with tf.io.TFRecordWriter('images.tfrecord') as writer:
    writer.write(example.SerializeToString())
