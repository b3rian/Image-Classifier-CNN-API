"""Test suite for the model preprocessing layer in TensorFlow."""
import pytest
import tensorflow as tf

from data.input_pipeline2 import get_datasets
from data.preprocessing import get_model_preprocessing_layer

# Configuration for the test
DATA_DIR = "D:/Downloads/tiny-224" 

@pytest.fixture(scope="session")
def train_ds():
    train_ds, _ = get_datasets(DATA_DIR, batch_size=8)
    return train_ds

@pytest.fixture
def preprocessing_layer():
    return get_model_preprocessing_layer()

def test_preprocessing_output_shape(train_ds, preprocessing_layer):
    for images, _ in train_ds.take(1):
        out = preprocessing_layer(images)
        assert out.shape == (8, 254, 254, 3), f"Expected shape (8, 254, 254, 3), got {out.shape}"

def test_preprocessing_value_range(train_ds, preprocessing_layer):
    for images, _ in train_ds.take(1):
        out = preprocessing_layer(images)
        min_val = tf.reduce_min(out).numpy()
        max_val = tf.reduce_max(out).numpy()
        assert min_val >= -3.0, f"Min value too low: {min_val}"
        assert max_val <= 3.0, f"Max value too high: {max_val}"

def test_preprocessing_no_nans(train_ds, preprocessing_layer):
    for images, _ in train_ds.take(1):
        out = preprocessing_layer(images)
        assert not tf.math.reduce_any(tf.math.is_nan(out)), "Output contains NaNs"
