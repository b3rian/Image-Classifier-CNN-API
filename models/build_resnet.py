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

    def call(self, x, training=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out += identity
        out = self.relu(out)

        return out
    
# ResNet class
class ResNet(Model):
    def __init__(self, block, layers_units, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = layers.Conv2D(64, kernel_size=7, strides=2,
                                   padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.maxpool = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')

        self.layer1 = self._make_layer(block, 64, layers_units[0])
        self.layer2 = self._make_layer(block, 128, layers_units[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers_units[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers_units[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def _make_layer(self, block, filters, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != filters * block.expansion:
            downsample = tf.keras.Sequential([
                layers.Conv2D(filters * block.expansion,
                              kernel_size=1, strides=stride, use_bias=False),
                layers.BatchNormalization()
            ])

        layers_list = []
        layers_list.append(block(filters, stride, downsample))
        self.in_channels = filters * block.expansion

        for _ in range(1, blocks):
            layers_list.append(block(filters))

        return tf.keras.Sequential(layers_list)
    
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

# Factory function for ResNet-18
def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
