import yaml
import tensorflow as tf
from data.input_pipeline import get_datasets
from utils.seed import set_seed
from utils.logger import logger
from trainers.trainer import trainer
from utils.metrics import get_classification_metrics
from models import build_vit, build_vgg_19, build_resnet
import os
from datetime import datetime

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_model(config):
    name = config["model"]["name"].lower()
    if name == "resnet":
        return resnet.build_resnet18()
    elif name == "vgg":
        return vgg.build_vgg19()
    elif name == "visiontransformer":
        return vit.build_vit_base()
    else:
        raise ValueError(f"Unknown model type: {name}")

def get_data(config):
    return get_datasets(config)


def get_callbacks(config):
    log_dir = os.path.join("logs", config["model"]["name"], datetime.now().strftime("%Y%m%d-%H%M%S"))

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join("checkpoints", config["model"]["name"] + ".h5"),
        save_best_only=True,
        monitor="val_accuracy"
    )

    return [tensorboard_cb, checkpoint_cb]

def main():
    config_path = "configs/resnet.yml"  # change to vit.yml or vgg.yml
    config = load_config(config_path)

    # Set seed
    set_seed(config.get("seed", 42))

    # Optional: Logger
    logger = setup_logger("main")
    logger.info(f"Loaded config from {config_path}")

    # Data
    train_ds, val_ds = get_data(config)

    # Model
    model = get_model(config)

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"]["initial"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=get_classification_metrics()
    )

    # Train
    trainer = Trainer(model, train_ds, val_ds, config, callbacks=get_callbacks(config))
    trainer.train()

if __name__ == "__main__":
    main()


 