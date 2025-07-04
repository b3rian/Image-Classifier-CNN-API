import tensorflow as tf
import os

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 200

def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img

def process_train_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = img / 255.0  # Normalize to [0,1]
    return img, tf.one_hot(label, NUM_CLASSES)

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