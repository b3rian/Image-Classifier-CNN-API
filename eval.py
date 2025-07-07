import tensorflow as tf
from utils.metrics import get_classification_metrics
from data.input_pipeline2 import get_datasets, get_test_dataset
import yaml
import argparse
import os

def evaluate(model_path, config):
     # Load test dataset (ignore train_ds and val_ds with `_, _`)
    _, _, test_ds = get_test_dataset()

    # load saved model without compiling
    model = tf.keras.models.load_model(model_path, compile=False)

    # recompile the model with the correct loss and metrics
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=get_classification_metrics()
    )
    # evaluate on the test dataset
    results = model.evaluate(test_ds, return_dict=True)
    print(f"/n[TEST RESULTS for {os.path.basename(model_path)}]")
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
