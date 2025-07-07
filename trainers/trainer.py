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
        self.strategy = self._init_strategy()

        # Wrap data loading and model creation inside strategy scope
        with self.strategy.scope():
            self.model = model_fn() # Create the model using the provided function
            self._compile_model() # Compile the model with optimizer, loss, and metrics
        
        self.train_ds = train_ds # Training dataset
        self.val_ds = val_ds # Validation dataset
        self.callbacks = get_callbacks(config=None) # Callbacks for training

    def _init_strategy(self):
        """Initialize TPU or default GPU /CPU strategy."""
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # Detect TPU
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu) # Initialize TPU strategy
            print("[INFO] Using TPU strategy.")
        except ValueError:
            strategy = tf.distribute.get_strategy()
            print("[INFO] Using default strategy (CPU/GPU).")
        return strategy

    # Compiling the model
    def _compile_model(self):
        """Compile the model with optimizer, loss, and metrics."""
        optimizer_cfg = self.config["training"]["optimizer"]
        opt_name = optimizer_cfg["name"].lower()
        lr = self.config["training"]["learning_rate"]["initial"]

        # configure optimizer based on the name
        if opt_name == "adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr,
                beta_1=optimizer_cfg.get("beta1", 0.9), # default beta1
                beta_2=optimizer_cfg.get("beta2", 0.999), # default beta2
                weight_decay=self.config["training"].get("weight_decay", 0.0) # default weight decay
            )
        elif opt_name == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr,
                momentum=optimizer_cfg.get("momentum", 0.9), # default momentum
                weight_decay=self.config["training"].get("weight_decay", 0.0) # default weight decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        # Compile the model with the optimizer, loss function, and metrics
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1),
            metrics=get_classification_metrics() # Get classification metrics (accuracy & top-5 accuracy)
        )
    
    # training loop method (to be called externally
    def train(self):
        """Run the training loop."""
        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.config["training"]["epochs"],
            verbose=1,
            callbacks=self.callbacks # Callbacks for logging, checkpointing, TensorBoard, etc.
        )
