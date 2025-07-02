import tensorflow as tf

# Data loader configuration for TensorFlow models
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 128
NUM_CLASSES = 1000

# decode a single example from the TFRecord file
def decode_example(serialized_example):
    """parse a single tf.train Example from TFRecord file"""
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.cast(example['image/class/label'], tf.int32) - 1  # Labels are 1-based
    return image, label
 

