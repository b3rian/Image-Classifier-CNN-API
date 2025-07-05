import tensorflow as tf

def get_model_preprocessing_layer():
    return tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255)  # Normalize pixels to [0, 1]
    ])
