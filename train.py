from trainers.trainer import Trainer
from models import build_resnet
from utils.logger import get_callbacks
from data.input_pipeline import get_datasets
from utils.seed import set_seed
import yaml
from models.resnet import ResNet18
from preprocessing import get_model_preprocessing_layer
import tensorflow as tf

config = yaml.safe_load(open("configs/resnet.yml"))
set_seed(config["seed"])

train_ds, val_ds = get_datasets(config)

def model_fn():
    preprocess = get_model_preprocessing_layer(config)
    
    inputs = tf.keras.Input(shape=(None, None, 3))
    x = preprocess(inputs)  # Apply preprocessing
    backbone = ResNet18(num_classes=config["model"]["num_classes"])
    outputs = backbone(x)

    return tf.keras.Model(inputs, outputs, name="resnet18_with_preprocessing")

trainer = Trainer(
    model_fn=build_resnet,
    train_ds=train_ds,
    val_ds=val_ds,
    config=config,
    callbacks=get_callbacks(config)
)

trainer.train()
