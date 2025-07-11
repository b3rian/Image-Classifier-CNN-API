import tensorflow as tf
from tensorflow.keras import layers, models

def simple_cnn_tiny_imagenet(input_shape=(64, 64, 3), num_classes=200):
    model = models.Sequential([
        # Layer 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        
        # Layer 2
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Layer 3
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),

        # Layer 4
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Layer 5
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),

        # Layer 6
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Flatten
        layers.Flatten(),

        # Layer 7 (Dense)
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),

        # Layer 8
        layers.Dense(256, activation='relu'),

        # Layer 9
        layers.Dense(128, activation='relu'),

        # Layer 10 - Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
