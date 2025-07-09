from tensorflow import keras
from tensorflow.keras import layers, initializers

def make_tiny_imagenet_model(input_shape=(64, 64, 3), num_classes=200):
    he_init = initializers.HeNormal()

    # Entry block
    x = layers.Conv2D(128, 3, strides=2, padding="same", kernel_initializer=he_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # For residual connection

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same", depthwise_initializer=he_init, pointwise_initializer=he_init)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same", depthwise_initializer=he_init, pointwise_initializer=he_init)(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same", kernel_initializer=he_init)(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x  # Set aside for next residual

    x = layers.SeparableConv2D(1024, 3, padding="same", depthwise_initializer=he_init, pointwise_initializer=he_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Output layer: 200 classes, softmax for probabilities
    outputs = layers.Dense(num_classes, activation="softmax", kernel_initializer=he_init)(x)

    return keras.Model(inputs, outputs)


 
