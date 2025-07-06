from trainers.trainer import Trainer
from models.resnet import build_resnet18
from utils.logger import get_callbacks
from input_pipeline import prepare_datasets
from utils.seed import set_seed
import yaml

config = yaml.safe_load(open("configs/resnet.yml"))
set_seed(config["seed"])

train_ds, val_ds = prepare_datasets(config)

trainer = Trainer(
    model_fn=build_resnet18,
    train_ds=train_ds,
    val_ds=val_ds,
    config=config,
    callbacks=get_callbacks(config)
)

trainer.train()
