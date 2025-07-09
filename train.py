import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled.")
    except RuntimeError as e:
        print(f"❌ Could not set memory growth: {e}")

from trainers.trainer import Trainer
from utils.logger import get_callbacks
from data.input_pipeline import get_datasets
from utils.seed import set_seed
import os
import yaml
from models.build_cnn_model import make_tiny_imagenet_model
from data.preprocessing import get_model_preprocessing_layer

 

# Load configurations
config = yaml.safe_load(open("configs/vgg19.yml"))

# Set random seed for reproducibility
set_seed(config["seed"])

data_dir = config["dataset"]["data_dir"]
batch_size = config["dataset"]["batch_size"]

# Get datasets
train_ds, val_ds = get_datasets(data_dir, batch_size)

# Adapt normalization layer
preprocess = get_model_preprocessing_layer()

# Model with preprocessing function
def model_fn():
    """Function to create the ResNet model with preprocessing."""
    inputs = tf.keras.Input(shape=(None, None, 3)) # Input layer for images
    x = preprocess(inputs)  # Apply preprocessing
    backbone = make_tiny_imagenet_model(num_classes=config["model"]["num_classes"])
    outputs = backbone(x)

    return tf.keras.Model(inputs, outputs, name="cnn_model_with_preprocessing")

# Initialize the Trainer with the model function and datasets
trainer = Trainer(
    model_fn=model_fn,
    train_ds=train_ds,
    val_ds=val_ds,
    config=config
)

# Start training
model = trainer.train()

# Save the trained model
os.makedirs("exports", exist_ok=True)
model.save("exports/vgg19_model.keras")
print("✅ Final model saved to exports/vgg19_model.keras")

