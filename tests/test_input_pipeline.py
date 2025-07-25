"""Unit tests for the input pipeline module."""
import pytest
import tensorflow as tf
import os

from data.input_pipeline2 import (
    get_label_map,
    load_dataset,
    create_dataset,
    get_datasets,
    get_test_dataset,
    IMAGENET_MEAN,
    NUM_CLASSES
)

# Test configuration for input pipeline
DATA_DIR = "D:/Downloads/tiny-224"

@pytest.fixture(scope="session")
def label_map():
    train_dir = os.path.join(DATA_DIR, "train")
    return get_label_map(train_dir)

@pytest.fixture(scope="session")
def train_val_datasets():
    return get_datasets(DATA_DIR, batch_size=8)  # Small batch for test

@pytest.fixture(scope="session")
def test_dataset(label_map):
    test_dir = os.path.join(DATA_DIR, "test")
    return get_test_dataset(test_dir, label_map, batch_size=8)

def test_train_dataset_shape(train_val_datasets):
    train_ds, _ = train_val_datasets
    for images, labels in train_ds.take(1):
        assert images.shape == (8, 224, 224, 3)
        assert labels.shape == (8, NUM_CLASSES)

def test_val_dataset_shape(train_val_datasets):
    _, val_ds = train_val_datasets
    for images, labels in val_ds.take(1):
        assert images.shape == (8, 224, 224, 3)
        assert labels.shape == (8, NUM_CLASSES)

def test_test_dataset_shape(test_dataset):
    for images, labels in test_dataset.take(1):
        assert images.shape == (8, 224, 224, 3)
        assert labels.shape == (8, NUM_CLASSES)

def test_image_pixel_range(train_val_datasets):
    train_ds, _ = train_val_datasets
    for images, _ in train_ds.take(1):
        min_val = tf.reduce_min(images)
        max_val = tf.reduce_max(images)
        assert min_val.numpy() >= -150  # After mean subtraction
        assert max_val.numpy() <= 300

def test_one_hot_labels(train_val_datasets):
    train_ds, _ = train_val_datasets
    for _, labels in train_ds.take(1):
        assert tf.reduce_all(tf.reduce_sum(labels, axis=-1) == 1.0)
