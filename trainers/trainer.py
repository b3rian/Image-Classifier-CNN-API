import tensorflow as tf
from utils.metrics import get_classification_metrics
from utils.logger import get_callbacks


class Trainer:
    def __init__(self, model_fn, train_ds, val_ds, config):
        """
        Args:
            model_fn (function): Function that returns a compiled or uncompiled Keras model.
            train_ds (tf.data.Dataset): Training dataset.
            val_ds (tf.data.Dataset): Validation dataset.
            config (dict): Configuration dictionary from YAML.
        """
        self.config = config
        self.model = model_fn()  # Create model directly (no strategy.scope)
        self._compile_model()    # Compile the model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.callbacks = get_callbacks()

    # Compiling the model
    def _compile_model(self):
        """Compile the model with optimizer, loss, and metrics."""
        optimizer_cfg = self.config["training"]["optimizer"]
        opt_name = optimizer_cfg["name"].lower()
        lr = self.config["training"]["learning_rate"]["initial"]

        if opt_name == "adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr,
                beta_1=optimizer_cfg.get("beta1", 0.9),
                beta_2=optimizer_cfg.get("beta2", 0.999),
                weight_decay=self.config["training"].get("weight_decay", 0.0)
            )
        elif opt_name == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr,
                momentum=optimizer_cfg.get("momentum", 0.9),
                weight_decay=self.config["training"].get("weight_decay", 0.0)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1),
            metrics=get_classification_metrics()
        )

    def train(self):
        """Run the training loop."""
        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.config["training"]["epochs"],
            verbose=1,
            callbacks=self.callbacks
        )
        return self.model
