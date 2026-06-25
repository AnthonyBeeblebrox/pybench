"""MNIST loading via the Hugging Face ``datasets`` library."""

from __future__ import annotations

import numpy as np
from datasets import Dataset, load_dataset


def _split_to_arrays(split: Dataset) -> tuple[np.ndarray, np.ndarray]:
    images = np.stack([np.asarray(img, dtype=np.float32) for img in split["image"]])
    x = images.reshape(len(images), -1) / 255.0
    y = np.asarray(split["label"], dtype=np.int32)
    return x, y


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST as flattened, normalized float arrays.

    Returns:
        A tuple ``(x_train, y_train, x_test, y_test)``. Images are ``float32``
        arrays of shape ``(n, 784)`` scaled to ``[0, 1]``; labels are ``int32``
        arrays of shape ``(n,)``.
    """
    ds = load_dataset("ylecun/mnist")
    x_train, y_train = _split_to_arrays(ds["train"])
    x_test, y_test = _split_to_arrays(ds["test"])
    return x_train, y_train, x_test, y_test
