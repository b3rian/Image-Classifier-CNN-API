import os
import yaml
from eval import evaluate

models = {
    "ResNet18": {
        "path": "checkpoints/resnet.h5",
        "config": "configs/resnet.yml"
    },
    "VGG19": {
        "path": "checkpoints/vgg19.h5",
        "config": "configs/vgg19.yml"
    },
    "ViT-Base": {
        "path": "checkpoints/vit_base.h5",
        "config": "configs/vit.yml"
    }
}

results = {}

for name, info in models.items():
    print(f"/n===== Evaluating {name} =====")
    config = yaml.safe_load(open(info["config"]))
    res = evaluate(info["path"], config)
    results[name] = res

# Summary comparison table
print("/n====== MODEL COMPARISON ======")
print(f"{'Model':<15} | {'Accuracy':<10} | {'Loss':<10}")
print("-" * 40)
for name, res in results.items():
    acc = res.get("sparse_categorical_accuracy", 0.0)
    loss = res.get("loss", 0.0)
    print(f"{name:<15} | {acc:.4f}     | {loss:.4f}")
