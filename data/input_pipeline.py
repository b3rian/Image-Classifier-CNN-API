import tensorflow as tf
import os

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 200

def parse_example(example_proto):
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed = tf.io.parse_single_example(example_proto, features)
    image = tf.io.decode_jpeg(parsed['image'], channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(parsed['label'], NUM_CLASSES)
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_crop(tf.image.resize_with_crop_or_pad(image, 256, 256), [IMAGE_SIZE, IMAGE_SIZE, 3])
    return image, label

def get_dataset(tfrecord_path, batch_size=128, training=True, cache=True, shuffle_buffer=2048, repeat=True):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        dataset = dataset.cache()

    if training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(shuffle_buffer)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset