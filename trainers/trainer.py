import tensorflow as tf
import os
from input_pipeline import get_datasets
from preprocessing import build_model  # assuming you have a model here

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