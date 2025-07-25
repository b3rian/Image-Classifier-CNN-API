import tensorflow as tf

# Utility function to get classification metrics for Keras models
def get_classification_metrics():
    return [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy")
    ]
