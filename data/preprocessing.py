import tensorflow as tf

def get_model_preprocessing_layer():
    return tf.keras.Sequential([
        tf.keras.layers.Resizing(224, 224),
        tf.keras.layers.Lambda(
            lambda img: tf.clip_by_value(img, 0.0, 255.0), name="clip_pixels"),
    tf.keras.layers.Rescaling(1./255)  # Normalize pixel values to [0, 1]
],  name="preprocessing_pipeline")  # Normalize using ImageNet statistics
