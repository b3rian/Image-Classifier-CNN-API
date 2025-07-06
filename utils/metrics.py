import tensorflow as tf

def get_classification_metrics():
    return [
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")
    ]
