import numpy as np
import cv2

from deepforensics.app import prnu


def test_extract_residual_shape_and_variance():
    # Synthetic BGR image with gradients
    h, w = 64, 96
    x = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    y = np.tile(np.linspace(0, 255, h, dtype=np.uint8), (w, 1)).T
    bgr = np.stack([x, y, (x//2 + y//2).astype(np.uint8)], axis=2)
    resid = prnu.extract_residual(bgr)
    assert resid.shape == (h, w)
    assert float(np.std(resid)) > 0.0

