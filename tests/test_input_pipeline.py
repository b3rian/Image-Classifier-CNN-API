import pytest
import tensorflow as tf
import input_pipeline
import os

DATA_DIR = "/content/drive/MyDrive/tiny-imagenet-200"

@pytest.fixture(scope="module")
def datasets():
    batch_size = 32
    train_ds, val_ds = input_pipeline.get_datasets(DATA_DIR, batch_size=batch_size)
    return train_ds, val_ds

def test_train_dataset_shape(datasets):
    train_ds, _ = datasets
    for images, labels in train_ds.take(1):
        assert images.shape[1:] == (64, 64, 3), "Image shape mismatch"
        assert labels.shape[1] == 200, "Label one-hot depth mismatch"

def test_val_dataset_shape(datasets):
    _, val_ds = datasets
    for images, labels in val_ds.take(1):
        assert images.shape[1:] == (64, 64, 3), "Image shape mismatch"
        assert labels.shape[1] == 200, "Label one-hot depth mismatch"

def test_pixel_range(datasets):
    train_ds, _ = datasets
    for images, _ in train_ds.take(1):
        assert tf.reduce_max(images) <= 1.0, "Image values exceed 1.0"
        assert tf.reduce_min(images) >= 0.0, "Image values below 0.0"
