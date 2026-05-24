"""Grad-CAM helpers for model interpretability."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


def find_last_conv_layer(model: keras.Model) -> keras.layers.Layer:
    """Find the last Conv2D layer in a Keras model.

    EfficientNet layer names can vary by TensorFlow/Keras version, so this
    function searches by layer type instead of relying on a hardcoded name.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer
        if isinstance(layer, keras.Model):
            try:
                return find_last_conv_layer(layer)
            except ValueError:
                continue
    raise ValueError(
        "Could not find a Conv2D layer automatically. "
        "Open model.summary() and pass a convolutional layer name manually."
    )


def make_gradcam_heatmap(
    image_batch: np.ndarray,
    model: keras.Model,
    last_conv_layer_name: Optional[str] = None,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """Create a Grad-CAM heatmap for one image batch.

    Grad-CAM visualizes which spatial regions of the fundus image contributed
    most strongly to the model prediction. This supports interpretability and
    helps check whether the classifier attends to clinically meaningful retinal
    regions rather than irrelevant artifacts.
    """
    if last_conv_layer_name:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    else:
        last_conv_layer = find_last_conv_layer(model)

    try:
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output],
        )
    except ValueError as exc:
        raise ValueError(
            "Grad-CAM could not connect the selected convolutional layer to the "
            "model output. Try printing model.summary() and pass a layer name "
            "from the main model graph."
        ) from exc

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_value + keras.backend.epsilon())
    return heatmap.numpy()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.40,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on an RGB image.

    image can be uint8 [0, 255] or float [0, 1]. The returned array is uint8.
    """
    if image.dtype != np.uint8:
        base = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        base = image.copy()

    heatmap_resized = cv2.resize(heatmap, (base.shape[1], base.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(base, 1.0 - alpha, colored, alpha, 0)
    return overlay


def save_overlay(image: np.ndarray, output_path: Path) -> None:
    """Save an RGB overlay image to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr)

