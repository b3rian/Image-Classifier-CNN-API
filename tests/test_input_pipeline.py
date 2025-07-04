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

     

 
