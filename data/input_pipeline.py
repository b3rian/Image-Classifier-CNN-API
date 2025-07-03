import tensorflow as tf

IMAGE_SIZE = 64
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