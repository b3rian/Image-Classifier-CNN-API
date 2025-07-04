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

def process_val_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    img = img / 255.0  # Normalize to [0,1]
    return img, tf.one_hot(label, NUM_CLASSES)

def get_label_map(train_dir):
    class_names = sorted(os.listdir(train_dir))
    label_map = {name: idx for idx, name in enumerate(class_names)}
    return label_map

def load_dataset(image_dir, label_map=None, is_training=True):
    image_paths = []
    labels = []

    if is_training:
        for class_name, class_index in label_map.items():
            class_dir = os.path.join(image_dir, class_name, "images")
            for fname in os.listdir(class_dir):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_index)
    else:
        val_img_dir = os.path.join(image_dir, "images")
        val_annotations = os.path.join(os.path.dirname(image_dir), "val_annotations.txt")
        with open(val_annotations, 'r') as f:
            for line in f:
                fname, class_name, *_ = line.strip().split()
                if class_name in label_map:
                    image_paths.append(os.path.join(val_img_dir, fname))
                    labels.append(label_map[class_name])

    return image_paths, labels

def create_dataset(image_paths, labels, batch_size=64, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
        dataset = dataset.map(process_train_image, num_parallel_calls=AUTOTUNE)
    else:
        dataset = dataset.map(process_val_image, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset