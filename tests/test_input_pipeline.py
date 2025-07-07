import pytest
import tensorflow as tf
import os

from data.input_pipeline2 import get_label_map, load_dataset, create_dataset

@pytest.fixture(scope="session")
def data_root():
    # Set this to your tiny-imagenet-200 root directory
    return "D:/Downloads/tiny-224" 

@pytest.fixture(scope="session")
def label_map(data_root):
    return get_label_map(os.path.join(data_root, "train"))

def test_train_dataset(data_root, label_map):
    train_dir = os.path.join(data_root, "train")
    paths, labels = load_dataset(train_dir, label_map, split="train")
    ds = create_dataset(paths, labels, batch_size=4, split="train")

    for images, labels in ds.take(1):
        assert images.shape == (4, 224, 224, 3)
        assert labels.shape == (4, 200)
        assert tf.reduce_all(tf.math.is_finite(images))

def test_val_dataset(data_root, label_map):
    val_dir = os.path.join(data_root, "val")
    paths, labels = load_dataset(val_dir, label_map, split="val")
    ds = create_dataset(paths, labels, batch_size=4, split="val")

    for images, labels in ds.take(1):
        assert images.shape == (4, 224, 224, 3)
        assert labels.shape == (4, 200)
        assert tf.reduce_all(tf.math.is_finite(images))

def test_test_dataset(data_root, label_map):
    test_dir = os.path.join(data_root, "test")
    paths, labels = load_dataset(test_dir, label_map, split="test")
    ds = create_dataset(paths, labels, batch_size=4, split="test")

    for images, labels in ds.take(1):
        assert images.shape == (4, 224, 224, 3)
        assert labels.shape == (4, 200)
        assert tf.reduce_all(tf.math.is_finite(images))
