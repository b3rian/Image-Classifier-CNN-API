import tensorflow as tf
import os

# Set your target directory
save_path = "D:/Documents/models"
os.makedirs(save_path, exist_ok=True)

# Load pretrained ResNet50 model with ImageNet weights
model = tf.keras.applications.ResNet50(weights="imagenet")

# Save in .keras format
model_file = os.path.join(save_path, "resnet50_imagenet.keras")
model.save(model_file)

print(f"✅ ResNet50 model saved at: {model_file}")
