import tensorflow as tf
import os

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 200

def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img

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