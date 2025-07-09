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
from models.build_resnet import build_resnet50_from_scratch

# Load configurations
config = yaml.safe_load(open("configs/resnet.yml"))

# Set random seed for reproducibility
set_seed(config["seed"])

data_dir = config["dataset"]["data_dir"]
batch_size = config["dataset"]["batch_size"]

# Get datasets
train_ds, val_ds, test_ds = get_datasets(data_dir, batch_size)

def model_fn():
    return build_resnet50_from_scratch(
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
model.save("exports/resnet50_model.keras")
print("✅ Final model saved to exports/resnet_model.keras")

