import tensorflow as tf
from tensorflow.keras import layers, models

def make_simple_cnn(input_shape=(64, 64, 3), num_classes=200):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

       

      


