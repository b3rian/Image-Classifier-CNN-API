import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 200

# This function decodes JPEG images to RGB format.
def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def process_train_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = decode_img(img)  # float32 in [0, 255]
    # Brightness & contrast (ImageNet-style color augmentation)
    img = tf.image.random_brightness(img, max_delta=25)  # delta in pixel range
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)

    return img, tf.one_hot(label, NUM_CLASSES)

def process_val_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = decode_img(img)  # Already resized to IMAGE_SIZE (64×64)


    return img, tf.one_hot(label, NUM_CLASSES)

def process_test_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    return img, tf.one_hot(label, NUM_CLASSES)

def get_label_map(train_dir):
    class_names = sorted(os.listdir(train_dir))
    label_map = {name: idx for idx, name in enumerate(class_names)}
    return label_map

def load_dataset(image_dir, label_map=None, split="train"):
    image_paths = []
    labels = []

    if split == "train":
        for class_name, class_index in label_map.items():
            class_dir = os.path.join(image_dir, class_name, "images")
            for fname in os.listdir(class_dir):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_index)

    elif split in ["val", "test"]:
        img_dir = os.path.join(image_dir, "images")
        annotation_file = os.path.join(image_dir, f"{split}_annotations.txt")

        with open(annotation_file, 'r') as f:
            for line in f:
                fname, class_name, *_ = line.strip().split()
                if class_name in label_map:
                    image_paths.append(os.path.join(img_dir, fname))
                    labels.append(label_map[class_name])
    else:
        raise ValueError(f"Unsupported split: {split}")

    return image_paths, labels

def create_dataset(image_paths, labels, batch_size=256, split="train"):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if split == "train":
        dataset = dataset.shuffle(buffer_size=len(image_paths))
        dataset = dataset.map(process_train_image, num_parallel_calls=AUTOTUNE)
    elif split == "val":
        dataset = dataset.map(process_val_image, num_parallel_calls=AUTOTUNE)
    elif split == "test":
        dataset = dataset.map(process_test_image, num_parallel_calls=AUTOTUNE)
    else:
        raise ValueError(f"Unsupported split: {split}")

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

def get_datasets(data_dir, batch_size=256, val_split=0.8):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    label_map = get_label_map(train_dir)

    # Load training data
    train_paths, train_labels = load_dataset(train_dir, label_map, split="train")

    # Load validation data (will split into val and test)
    val_paths, val_labels = load_dataset(val_dir, label_map, split="val")

    # Split val data into validation and test (80% val and 20% test)
    val_paths_split, test_paths_split, val_labels_split, test_labels_split = train_test_split(
        val_paths, val_labels, test_size=(1 - val_split), stratify=val_labels, random_state=42
    )

    # Create datasets
    train_ds = create_dataset(train_paths, train_labels, batch_size=batch_size, split="train")
    val_ds = create_dataset(val_paths_split, val_labels_split, batch_size=batch_size, split="val")
    test_ds = create_dataset(test_paths_split, test_labels_split, batch_size=batch_size, split="test")

    return train_ds, val_ds, test_ds


