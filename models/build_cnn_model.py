import tensorflow as tf
from tensorflow.keras import layers, models, initializers

def make_tiny_imagenet_model(input_shape=(None, None, 3), num_classes=200):
    he_init = tf.keras.initializers.HeNormal()

    inputs = tf.keras.Input(shape=input_shape)

    # ✅ Preprocessing block inside the model
    x = tf.keras.layers.Resizing(64, 64)(inputs)  # Ensure size is consistent
    x = tf.keras.layers.Rescaling(1. / 255)(x)     # [0, 255] → [0, 1]

    # Entry block
    x = layers.Conv2D(128, 3, strides=2, padding="same", kernel_initializer=he_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same",
                                   depthwise_initializer=he_init,
                                   pointwise_initializer=he_init)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same",
                                   depthwise_initializer=he_init,
                                   pointwise_initializer=he_init)(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same", kernel_initializer=he_init)(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same",
                               depthwise_initializer=he_init,
                               pointwise_initializer=he_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax", kernel_initializer=he_init)(x)

    return tf.keras.Model(inputs, outputs, name="cnn_model_with_preprocessing")
