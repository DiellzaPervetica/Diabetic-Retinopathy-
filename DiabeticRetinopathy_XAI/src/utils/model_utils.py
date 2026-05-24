"""Model builders, class weights, callbacks, and training plots."""

import os
import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    CLASS_NAMES,
    IMAGE_SHAPE,
    LEARNING_RATE,
    NUM_CLASSES,
    RANDOM_SEED,
)


def set_global_determinism(seed: int = RANDOM_SEED) -> None:
    """Set common random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_data_augmentation() -> keras.Sequential:
    """Return lightweight augmentation layers used only during training."""
    augmentation_layers = [
        layers.RandomFlip("horizontal", name="random_horizontal_flip"),
        layers.RandomRotation(0.05, name="random_rotation"),
        layers.RandomZoom(0.10, name="random_zoom"),
        layers.RandomContrast(0.10, name="random_contrast"),
    ]

    # RandomBrightness is available in newer TensorFlow/Keras versions.
    if hasattr(layers, "RandomBrightness"):
        augmentation_layers.append(
            layers.RandomBrightness(0.10, value_range=(0.0, 1.0), name="random_brightness")
        )

    return keras.Sequential(augmentation_layers, name="data_augmentation")


def build_baseline_cnn(
    input_shape: tuple[int, int, int] = IMAGE_SHAPE,
    num_classes: int = NUM_CLASSES,
    use_augmentation: bool = True,
) -> keras.Model:
    """Build a deliberately simple baseline CNN."""
    inputs = keras.Input(shape=input_shape, name="image")
    x = build_data_augmentation()(inputs) if use_augmentation else inputs

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")


def build_efficientnetb0(
    input_shape: tuple[int, int, int] = IMAGE_SHAPE,
    num_classes: int = NUM_CLASSES,
    weights: Optional[str] = "imagenet",
    train_base: bool = False,
    dropout_rate: float = 0.35,
    use_augmentation: bool = True,
) -> keras.Model:
    """Build EfficientNetB0 transfer learning model.

    The tf.data pipeline returns images normalized to [0, 1]. Keras EfficientNet
    models with ImageNet weights expect the standard EfficientNet preprocessing
    path, so a Rescaling(255.0) layer converts the model input back to [0, 255]
    before the pretrained backbone.
    """
    inputs = keras.Input(shape=input_shape, name="image")
    x = build_data_augmentation()(inputs) if use_augmentation else inputs
    x = layers.Rescaling(255.0, name="to_0_255_for_efficientnet")(x)

    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_tensor=x,
    )
    base_model.trainable = train_base

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(dropout_rate, name="classification_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="efficientnetb0_dr")


def compile_model(
    model: keras.Model,
    learning_rate: float = LEARNING_RATE,
    loss: str = "sparse_categorical_crossentropy",
    focal_gamma: float = 2.0,
) -> keras.Model:
    """Compile a multiclass classifier."""
    if loss == "sparse_focal_loss":
        loss_fn = sparse_categorical_focal_loss(gamma=focal_gamma)
    else:
        loss_fn = "sparse_categorical_crossentropy"

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model


def sparse_categorical_focal_loss(gamma: float = 2.0, alpha: Optional[np.ndarray] = None):
    """Return sparse categorical focal loss for imbalanced multiclass data."""

    def loss_fn(y_true, y_pred):
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred_clipped = tf.clip_by_value(
            y_pred,
            keras.backend.epsilon(),
            1.0 - keras.backend.epsilon(),
        )
        y_true_one_hot = tf.one_hot(y_true_int, depth=tf.shape(y_pred_clipped)[-1])
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred_clipped)
        focal_factor = tf.pow(1.0 - y_pred_clipped, gamma)
        loss = focal_factor * cross_entropy
        if alpha is not None:
            alpha_tensor = tf.constant(alpha, dtype=tf.float32)
            loss = loss * alpha_tensor
        return tf.reduce_sum(loss, axis=-1)

    return loss_fn


def compute_balanced_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Compute class weights to reduce the effect of class imbalance."""
    labels = np.asarray(labels).astype(int)
    present_classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=labels,
    )
    class_weights = {int(cls): float(weight) for cls, weight in zip(present_classes, weights)}

    # Keras accepts missing classes, but filling all labels makes logs clearer.
    for label in CLASS_NAMES:
        class_weights.setdefault(label, 1.0)
    return class_weights


def build_callbacks(
    model_path: Path,
    monitor: str = "val_loss",
    patience: int = 4,
) -> list[keras.callbacks.Callback]:
    """Create standard training callbacks."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def history_to_dataframe(history: keras.callbacks.History, phase: str = "training") -> pd.DataFrame:
    """Convert a Keras History object to a dataframe."""
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.insert(0, "phase", phase)
    return history_df


def save_history(history_df: pd.DataFrame, output_path: Path) -> None:
    """Save training history to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(output_path, index=False)


def plot_training_curves(history_df: pd.DataFrame, output_path: Path, title: str) -> None:
    """Save accuracy and loss curves from a history dataframe."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    accuracy_column = "accuracy" if "accuracy" in history_df.columns else "binary_accuracy"
    val_accuracy_column = (
        "val_accuracy" if "val_accuracy" in history_df.columns else "val_binary_accuracy"
    )

    for phase, group in history_df.groupby("phase"):
        if accuracy_column in group:
            axes[0].plot(
                group["epoch"],
                group[accuracy_column],
                marker="o",
                label=f"{phase} train",
            )
        if val_accuracy_column in group:
            axes[0].plot(
                group["epoch"],
                group[val_accuracy_column],
                marker="o",
                linestyle="--",
                label=f"{phase} val",
            )

        axes[1].plot(group["epoch"], group["loss"], marker="o", label=f"{phase} train")
        if "val_loss" in group:
            axes[1].plot(
                group["epoch"],
                group["val_loss"],
                marker="o",
                linestyle="--",
                label=f"{phase} val",
            )

    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_class_weights(class_weights: dict[int, float]) -> None:
    """Pretty-print class weights."""
    print("\nClass weights used during training:")
    for label in sorted(class_weights):
        print(f"  {label} ({CLASS_NAMES[label]}): {class_weights[label]:.3f}")
