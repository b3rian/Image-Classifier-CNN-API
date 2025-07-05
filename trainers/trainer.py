import tensorflow as tf
import os
from input_pipeline import get_datasets
from preprocessing import build_model

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
DATA_DIR = "/path/to/tiny-imagenet-200"  # CHANGE THIS TO YOUR DATA PATH
BATCH_SIZE = 128 * strategy.num_replicas_in_sync
EPOCHS = 10

# Load datasets
train_ds, val_ds = get_datasets(data_dir=DATA_DIR, batch_size=BATCH_SIZE)