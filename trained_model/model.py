import tensorflow
from tensorflow.keras.applications import EfficientNetV2L
model = EfficientNetV2L(weights="imagenet")
model.save("custom_cnn_model_1000_classes.keras")
