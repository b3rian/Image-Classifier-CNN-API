import random
import numpy as np
import tensorflow as tf
import os

def set_seed(seed: int = 42):
    """
    Set seed for Python, NumPy, and TensorFlow to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # For even more reproducibility
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
