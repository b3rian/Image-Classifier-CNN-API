import tensorflow as tf
from tensorflow.keras import layers, models

def conv3x3(filters, stride=1):
    return layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False,
                         kernel_initializer='he_normal')

def basic_block(x, filters, stride=1, downsample=False):
    identity = x

    out = conv3x3(filters, stride)(x)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)

    out = conv3x3(filters)(out)
    out = layers.BatchNormalization()(out)

    if downsample:
        identity = layers.Conv2D(filters, kernel_size=1, strides=stride, use_bias=False,
                                 kernel_initializer='he_normal')(identity)
        identity = layers.BatchNormalization()(identity)

    out = layers.Add()([out, identity])
    out = layers.ReLU()(out)

    return out

def make_layer(x, filters, blocks, stride):
    x = basic_block(x, filters, stride=stride, downsample=True)
    for _ in range(1, blocks):
        x = basic_block(x, filters)
    return x

def ResNet18(input_shape=(224, 224, 3), num_classes=1000):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False,
                      kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = make_layer(x, filters=64,  blocks=2, stride=1)
    x = make_layer(x, filters=128, blocks=2, stride=2)
    x = make_layer(x, filters=256, blocks=2, stride=2)
    x = make_layer(x, filters=512, blocks=2, stride=2)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="ResNet18")
    return model

 
     