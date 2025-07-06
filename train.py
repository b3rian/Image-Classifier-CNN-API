from trainers.trainer import Trainer
from models import build_resnet
from utils.logger import get_callbacks
from data.input_pipeline import get_datasets
from utils.seed import set_seed
import yaml

config = yaml.safe_load(open("configs/resnet.yml"))
set_seed(config["seed"])

train_ds, val_ds = get_datasets(config)

trainer = Trainer(
    model_fn=build_resnet,
    train_ds=train_ds,
    val_ds=val_ds,
    config=config,
    callbacks=get_callbacks(config)
)

trainer.train()
