import os
from datetime import datetime
import tensorflow as tf

def get_callbacks(log_dir_base="logs", checkpoint_dir="checkpoints", monitor="val_loss"):
    """
    Creates and returns a list of commonly used Keras callbacks.

    Args:
        log_dir_base (str): Base directory for TensorBoard logs.
        checkpoint_dir (str): Directory to save model checkpoints.
        monitor (str): Metric to monitor for checkpointing and early stopping.

    Returns:
        List[tf.keras.callbacks.Callback]
    """
    # Timestamped log folder
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_dir_base, timestamp)
    ckpt_path = os.path.join(checkpoint_dir, f"best_model_{timestamp}.h5")

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.1,
            patience=3,
            verbose=1
        )
    ]
