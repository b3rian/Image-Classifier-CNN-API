import tensorflow as tf
from tensorflow.keras import layers, models

def VGG19(input_shape=(224, 224, 3), num_classes=1000):
    model = models.Sequential()