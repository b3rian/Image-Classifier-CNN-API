import tensorflow as tf
from trainers.trainer import Trainer
from utils.logger import get_callbacks
from data.input_pipeline import get_datasets
from utils.seed import set_seed
import os
import yaml
from models.build_vgg_19 import simple_cnn_tiny_imagenet

# Load configurations
config = yaml.safe_load(open("configs/resnet.yml"))

# Set random seed for reproducibility
set_seed(config["seed"])

data_dir = config["dataset"]["data_dir"]
batch_size = config["dataset"]["batch_size"]

# Get datasets
train_ds, val_ds, test_ds = get_datasets(data_dir, batch_size)

def model_fn():
    return simple_cnn_tiny_imagenet(
        input_shape=(64, 64, 3),
        num_classes=config["model"]["num_classes"]
    )

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
model.save("exports/custom_model.keras")
print("✅ Final model saved to exports/custom_model.keras")

