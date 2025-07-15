import tensorflow as tf

# Loading the Keras model
model = tf.keras.models.load_model("D:\Telegram Desktop\custom_cnn_model_1000_classes.keras")

# Convert to TFLite with float16 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Save to desired path
output_path = r"D:\Documents\Datasets\custom_model.tflite"
with open(output_path, "wb") as f:
    f.write(tflite_model)

print("Model converted and saved as custom_model.tflite in D:\\Documents\\Datasets")

