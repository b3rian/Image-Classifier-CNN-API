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
