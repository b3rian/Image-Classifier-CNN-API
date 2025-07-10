import tensorflow as tf
from tensorflow.keras import layers, models

def SimpleCNN(input_shape=(32, 32, 3), num_classes=10):
    """
    A simple CNN model for image classification.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
    
    Returns:
        A Keras Model instance
    """
    model = models.Sequential([
        # Convolutional Base
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Classifier Head
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Optional: Add a function to compile the model
def create_and_compile_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Creates and compiles the SimpleCNN model with default settings
    """
    model = SimpleCNN(input_shape, num_classes)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
     