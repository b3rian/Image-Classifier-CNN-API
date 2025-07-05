import tensorflow as tf
from tensorflow.keras import layers, Model

# Residual block (BasicBlock as in ResNet-18 and 34)
class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, filters, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size=3, strides=stride,
                                   padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, kernel_size=3, strides=1,
                                   padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.downsample = downsample