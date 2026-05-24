"""Image loading and lightweight preprocessing utilities."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from config import IMAGE_SIZE


def load_image_pil(
    image_path: Path,
    image_size: Optional[Tuple[int, int]] = (IMAGE_SIZE, IMAGE_SIZE),
    normalize: bool = True,
) -> np.ndarray:
    """Load an image with Pillow, convert to RGB, resize, and optionally normalize.

    Parameters
    ----------
    image_path:
        Path to the fundus image.
    image_size:
        Target size as (height, width). If None, original dimensions are kept.
    normalize:
        If True, return float32 pixels in [0, 1]. Otherwise return uint8 pixels.
    """
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if image_size is not None:
            img = img.resize((image_size[1], image_size[0]))
        array = np.asarray(img)

    if normalize:
        return array.astype("float32") / 255.0
    return array


def read_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """Return image dimensions as (width, height) without loading all pixels."""
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        return img.size


def load_and_preprocess_image_tf(
    image_path,
    label,
    image_size: int = IMAGE_SIZE,
):
    """TensorFlow image loader for tf.data pipelines.

    The returned image is RGB, resized to image_size x image_size, and normalized
    to float32 pixels in [0, 1]. This function is memory efficient because each
    image is decoded only when a batch is requested.
    """
    import tensorflow as tf

    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def optional_crop_black_borders(image: np.ndarray, threshold: int = 7) -> np.ndarray:
    """Optionally crop dark borders from a fundus image.

    This is not used by default because the project aims to keep preprocessing
    simple and stable. It can be useful for experiments where black borders take
    up a large part of the image.
    """
    import cv2

    if image.dtype != np.uint8:
        work = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        work = image.copy()

    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image[y0:y1, x0:x1]


def optional_apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Optionally apply CLAHE contrast enhancement to an RGB image.

    CLAHE can improve local contrast in retinal images, but it also changes the
    visual distribution of the data. For a clean methodology project, keep it as
    an optional experiment rather than default preprocessing.
    """
    import cv2

    if image.dtype != np.uint8:
        work = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        work = image.copy()

    lab = cv2.cvtColor(work, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    if image.dtype != np.uint8:
        return enhanced.astype("float32") / 255.0
    return enhanced


def preprocess_fundus_image(
    image_path: Path,
    output_path: Path,
    image_size: int = IMAGE_SIZE,
    crop_black_borders: bool = True,
    apply_clahe: bool = True,
    jpeg_quality: int = 95,
) -> Path:
    """Create a lightweight preprocessed fundus image on disk.

    The preprocessing is intentionally conservative:
    - RGB conversion
    - optional black-border crop
    - optional CLAHE contrast enhancement
    - resize to image_size x image_size

    Saving 224x224 images makes later training faster and keeps memory use low.
    """
    import cv2

    image = load_image_pil(Path(image_path), image_size=None, normalize=False)
    if crop_black_borders:
        image = optional_crop_black_borders(image)
    if apply_clahe:
        image = optional_apply_clahe(image)

    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return output_path
