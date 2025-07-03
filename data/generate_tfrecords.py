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

def build_class_index_map(class_list_file):
    with open(class_list_file) as f:
        classes = [line.strip() for line in f]
    return {cls_name: idx for idx, cls_name in enumerate(classes)}

def write_tfrecord(images_dir, output_file, class_index_map, val_annotations=None):
    with tf.io.TFRecordWriter(output_file) as writer:
        image_paths = glob(os.path.join(images_dir, '*.JPEG'))
        for path in image_paths:
            try:
                image_data = tf.io.read_file(path)
                filename = os.path.basename(path)

                # Get label from directory name or annotation file
                if val_annotations:
                    class_name = val_annotations[filename]
                else:
                    class_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
                
                label = class_index_map[class_name]
                tf_example = image_example(image_data.numpy(), label)
                writer.write(tf_example.SerializeToString())
            except Exception as e:
                print(f"Skipping {path}: {e}")
     
