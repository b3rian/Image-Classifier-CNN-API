import tensorflow as tf

def create_example(image, label):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


with tf.io.TFRecordWriter("data.tfrecord") as writer:
    for img, label in dataset:  # assume dataset is a list of (image, label)
        img_raw = img.tobytes()
        example = create_example(img_raw, label)
        writer.write(example.SerializeToString())

feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
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
