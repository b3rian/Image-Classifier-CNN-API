import tensorflow as tf

# Data loader configuration for TensorFlow models
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = 224
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
 
# image augmentation function
def augment_image(image, label):
    image = tf.image.resize(image, [IMAGE_SIZE + 32, IMAGE_SIZE + 32])
    image = tf.image.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, label

def preprocess_image(image, label):
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    return image, label

# dataset builder function
def build_dataset(filenames, training=True):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    
    if training:
        dataset = dataset.shuffle(buffer_size=10000)
    
    dataset = dataset.map(decode_example, num_parallel_calls=AUTOTUNE)
    
    if training:
        dataset = dataset.map(augment_image, num_parallel_calls=AUTOTUNE)
    else:
        dataset = dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

TRAIN_FILENAMES = tf.io.gfile.glob("gs://imagenet-cils/TFRs/train/*.tfrecord")
VAL_FILENAMES = tf.io.gfile.glob("gs://imagenet-cils/TFRs/validation/*.tfrecord")

train_ds = build_dataset(TRAIN_FILENAMES, training=True)
val_ds = build_dataset(VAL_FILENAMES, training=False)
