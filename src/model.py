import json
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard
)
from pathlib import Path


def create_cnn_model(input_shape, num_classes, learning_rate=0.001):
    """
    Create a CNN model for satellite image classification.
    
    Architecture:
    - 4 Convolutional blocks with increasing filters (32, 64, 128, 256)
    - Each block: Conv2D -> BatchNorm -> ReLU -> MaxPooling -> Dropout
    - Global Average Pooling
    - Dense layers with dropout for classification
    """
    
    model = models.Sequential(name="SatelliteCNN")
    
    # Input layer
    model.add(layers.Input(shape=input_shape))
    
    # Block 1: 32 filters
    model.add(layers.Conv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Block 2: 64 filters
    model.add(layers.Conv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Block 3: 128 filters
    model.add(layers.Conv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    # Block 4: 256 filters
    model.add(layers.Conv2D(256, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    # Global Average Pooling
    model.add(layers.GlobalAveragePooling2D())
    
    # Dense layers
    model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )
    
    return model


def create_lightweight_cnn(input_shape, num_classes, learning_rate=0.001):
    """
    Create a lighter CNN model for faster training/inference.
    """
    model = models.Sequential(name="SatelliteCNN_Light")
    
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(config, model_save_path):
    """Create training callbacks."""
    callbacks = []
    
    # Early stopping
    callbacks.append(EarlyStopping(
        monitor='val_loss',
        patience=config['training']['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    ))
    
    # Reduce learning rate
    callbacks.append(ReduceLROnPlateau(
        monitor='val_loss',
        factor=config['training']['reduce_lr_factor'],
        patience=config['training']['reduce_lr_patience'],
        min_lr=config['training']['min_lr'],
        verbose=1
    ))
    
    # Model checkpoint
    callbacks.append(ModelCheckpoint(
        filepath=str(model_save_path),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ))
    
    # TensorBoard
    log_dir = Path("logs") / "fit"
    log_dir.mkdir(parents=True, exist_ok=True)
    callbacks.append(TensorBoard(log_dir=str(log_dir), histogram_freq=1))
    
    return callbacks


def save_model_architecture(model, json_path):
    """Save model architecture to JSON."""
    model_json = model.to_json()
    with open(json_path, 'w') as f:
        f.write(model_json)
    print(f"Model architecture saved to {json_path}")


def load_model_architecture(json_path):
    """Load model architecture from JSON."""
    with open(json_path, 'r') as f:
        model_json = f.read()
    model = tf.keras.models.model_from_json(model_json)
    return model


def get_model_summary(model):
    """Get formatted model summary."""
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return '\n'.join(summary_lines)


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = total_params - trainable_params
    return total_params, trainable_params, non_trainable_params