import os
from datetime import datetime
import tensorflow as tf

def get_callbacks(base_dir="experiments", monitor="val_loss"):
    """
    Creates and returns a list of commonly used Keras callbacks with proper experiment logging.

    Args:
        base_dir (str): Base directory to store logs, checkpoints, and metrics.
        monitor (str): Metric to monitor for checkpointing, LR reduction, and early stopping.

    Returns:
        List[tf.keras.callbacks.Callback]
    """
    # Timestamp for experiment versioning
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = os.path.join(base_dir, timestamp)

    # Subdirectories
    log_dir = os.path.join(experiment_dir, "logs")
    ckpt_path = os.path.join(experiment_dir, "checkpoints", "best_resnet18_model.keras")
    csv_log_path = os.path.join(experiment_dir, "metrics.csv")

    # Ensure directories exist
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    return [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=csv_log_path,
            append=False
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.1,
            patience=3,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=6,
            restore_best_weights=True,
            verbose=1
        )
    ]
