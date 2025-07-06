import tensorflow as tf
import os
from data.input_pipeline import get_datasets
from data.preprocessing import build_model
from utils.logger import get_callbacks
from utils.seed import set_seed

set_seed(1234)


# Initialize TPU strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # Detect TPU
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("✅ TPU detected and initialized")
except ValueError:
    strategy = tf.distribute.get_strategy()  # fallback to default (CPU/GPU)
    print("❌ TPU not found. Using default strategy")

# Set paths and batch size
DATA_DIR = "/path/to/tiny-imagenet-200" 
BATCH_SIZE = 128 * strategy.num_replicas_in_sync
EPOCHS = 10
CHECKPOINT_PATH = "checkpoints/best_model.h5"
SAVED_MODEL_DIR = "saved_model/my_model"

# Load datasets
train_ds, val_ds = get_datasets(data_dir=DATA_DIR, batch_size=BATCH_SIZE)
callbacks = get_callbacks(
    log_dir_base=f"logs/{model_name}",
    checkpoint_dir=f"checkpoints/{model_name}",
    monitor="val_loss"
)


# Model definition inside strategy.scope()
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(200, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

# Train the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb]
)