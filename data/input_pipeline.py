import tensorflow as tf
import os

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 200
# ImageNet mean (RGB)
IMAGENET_MEAN = tf.constant([123.675, 116.28, 103.53], dtype=tf.float32)  # [R, G, B]

def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img

def process_train_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = decode_img(img)  # float32 in [0, 255]

    # 1. Resize shorter side randomly between 256 and 480
    rand_size = tf.random.uniform([], minval=256, maxval=480, dtype=tf.int32)
    shape = tf.shape(img)[:2]
    h, w = shape[0], shape[1]
    scale = tf.cast(rand_size, tf.float32) / tf.cast(tf.minimum(h, w), tf.float32)
    new_height = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
    new_width = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)
    img = tf.image.resize(img, [new_height, new_width])

    # Random crop to 224×224
    img = tf.image.random_crop(img, size=[224, 224, 3])

    # Random horizontal flip
    img = tf.image.random_flip_left_right(img)

    # Brightness & contrast (ImageNet-style color augmentation)
    img = tf.image.random_brightness(img, max_delta=25)  # delta in pixel range
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)

    # Subtract per-channel ImageNet mean
    img = img - IMAGENET_MEAN

    return img, tf.one_hot(label, NUM_CLASSES)

def process_val_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = decode_img(img)  # Already resized to IMAGE_SIZE (64×64)

    # Resize shorter side to 256 (ImageNet-style)
    target_shorter_side = 256
    shape = tf.shape(img)[:2]
    h, w = shape[0], shape[1]
    scale = tf.cast(target_shorter_side, tf.float32) / tf.cast(tf.minimum(h, w), tf.float32)
    new_height = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
    new_width = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)
    img = tf.image.resize(img, [new_height, new_width])

    # Center crop to 224×224
    img = tf.image.resize_with_crop_or_pad(img, 224, 224)

    # Subtract mean
    img = img - IMAGENET_MEAN

    return img, tf.one_hot(label, NUM_CLASSES)

def process_test_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    # Resize shorter side to 256
    target_shorter_side = 256
    shape = tf.shape(img)[:2]
    h, w = shape[0], shape[1]
    scale = tf.cast(target_shorter_side, tf.float32) / tf.cast(tf.minimum(h, w), tf.float32)
    new_height = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
    new_width = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)
    img = tf.image.resize(img, [new_height, new_width])

    # Center crop to 224×224
    img = tf.image.resize_with_crop_or_pad(img, 224, 224)

    # Subtract mean
    img = img - IMAGENET_MEAN

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

def create_dataset(image_paths, labels, batch_size=64, split="train"):
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


def get_datasets(data_dir, batch_size=64):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    label_map = get_label_map(train_dir)

    train_paths, train_labels = load_dataset(train_dir, label_map, split="train")
    val_paths, val_labels = load_dataset(val_dir, label_map, split="val")

    train_ds = create_dataset(train_paths, train_labels, batch_size=batch_size, split="train")
    val_ds = create_dataset(val_paths, val_labels, batch_size=batch_size, split="val")

    return train_ds, val_ds


def get_test_dataset(test_dir, label_map, batch_size=64):
    test_paths, test_labels = load_dataset(test_dir, label_map, is_training=False)
    test_ds = create_dataset(test_paths, test_labels, batch_size=batch_size, split="test")
    return test_ds
