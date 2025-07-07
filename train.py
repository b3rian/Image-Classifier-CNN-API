from trainers.trainer import Trainer
from utils.logger import get_callbacks
from data.input_pipeline2 import get_datasets
from utils.seed import set_seed
import yaml
from models.build_simple_cnn import SimpleCNN
from data.preprocessing import get_model_preprocessing_layer
import tensorflow as tf

# Load configurations
config = yaml.safe_load(open("configs/resnet.yml"))

# Set random seed for reproducibility
set_seed(config["seed"])

data_dir = config["dataset"]["data_dir"]
batch_size = config["dataset"]["batch_size"]

# Get datasets
train_ds, val_ds = get_datasets(data_dir, batch_size)

# Adapt normalization layer
preprocess = get_model_preprocessing_layer()
for layer in preprocess.layers:
    if isinstance(layer, tf.keras.layers.Normalization):
        print("[INFO] Adapting normalization layer...")

        # Extract only image tensors and take a limited number of batches to adapt
        image_ds = train_ds.map(lambda x, y: x).unbatch().take(1000)  # unbatch is key
        image_ds = image_ds.map(lambda x: tf.cast(x, tf.float32))
        layer.adapt(image_ds)

# Model with preprocessing function
def model_fn():
    """Function to create the ResNet model with preprocessing."""
    inputs = tf.keras.Input(shape=(None, None, 3)) # Input layer for images
    x = preprocess(inputs)  # Apply preprocessing
    backbone = SimpleCNN(num_classes=config["model"]["num_classes"])
    outputs = backbone(x)

    return tf.keras.Model(inputs, outputs, name="resnet18_with_preprocessing")

# Initialize the Trainer with the model function and datasets
trainer = Trainer(
    model_fn=model_fn,
    train_ds=train_ds,
    val_ds=val_ds,
    config=config,
    callbacks=get_callbacks(config)
)

# Start training
trainer.train()
