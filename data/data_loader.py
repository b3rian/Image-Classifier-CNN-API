import tensorflow as tf
import os
import csv
from glob import glob

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_string, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(image_string),
        'label': _int64_feature(label),
    }))

 
def get_val_labels(val_annotations_path):
    label_map = {}
    with open(val_annotations_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            label_map[row[0]] = row[1]
    return label_map

# resizing after training
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
