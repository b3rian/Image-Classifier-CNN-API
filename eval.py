import tensorflow as tf
from metrics.metrics import get_classification_metrics
from input_pipeline import prepare_datasets
import yaml
import argparse
import os

def evaluate(model_path, config):
    _, val_ds = prepare_datasets(config)

    model = tf.keras.models.load_model(model_path, compile=False)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=get_classification_metrics()
    )

    results = model.evaluate(val_ds, return_dict=True)
    print(f"/n[RESULTS for {os.path.basename(model_path)}]")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .h5 or SavedModel")
    parser.add_argument("--config", type=str, default="configs/resnet.yml", help="Path to config YAML")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    evaluate(args.model_path, config)
