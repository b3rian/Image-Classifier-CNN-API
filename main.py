import tensorflow as tf
import os
from input_pipeline import get_dataset
from utils.seed import set_seed

set_seed(1234)


# Set up TPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy()

train_ds = get_dataset("/content/train.tfrecord", training=True, batch_size=128)
val_ds = get_dataset("/content/val.tfrecord", training=False, batch_size=128, repeat=False)

# Define model inside strategy scope
with strategy.scope():
    model = tf.keras.applications.ResNet50(  # or tf.keras.models.load_model() for ViT
        weights=None,
        input_shape=(224, 224, 3),
        classes=200
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_ds, epochs=30, validation_data=val_ds, steps_per_epoch=1000, validation_steps=100)