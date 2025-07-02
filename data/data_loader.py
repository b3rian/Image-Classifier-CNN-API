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
    parsed = tf.io.parse_single_example(proto, feature_description)
    parsed['image'] = tf.io.decode_raw(parsed['image'], tf.uint8)
    return parsed['image'], parsed['label']

raw_dataset = tf.data.TFRecordDataset("data.tfrecord")
parsed_dataset = raw_dataset.map(_parse_function)

for image, label in parsed_dataset.take(1):
    print(image.shape, label.numpy())

