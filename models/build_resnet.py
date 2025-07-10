import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

def build_resnet50_from_scratch(input_shape=(64, 64, 3), num_classes=200):
    base_model = ResNet50(
        include_top=False,
        weights=None,  # <--- Train from scratch
        input_shape=input_shape,
        pooling='avg'
    )

    # Wrap with classification head
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=True)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model
     