__all__ = ["COLORS", "CELL_PX", "WALL_PX", "tile_images", "draw_text_at", "resize"]

from typing import Sequence, Tuple

import cv2
import numpy as np

# Pixel constants shared across envs
CELL_PX = 16
WALL_PX = int(CELL_PX / 4)

# Medium color palette used by all environments
class COLORS:
    black = np.array([0, 0, 0], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)
    gray = np.array([128, 128, 128], dtype=np.uint8)
    gray_light = np.array([191, 191, 191], dtype=np.uint8)
    gray_mid_light = np.array([160, 160, 160], dtype=np.uint8)
    gray_dark = np.array([64, 64, 64], dtype=np.uint8)
    gray_mid_dark = np.array([96, 96, 96], dtype=np.uint8)
    red = np.array([128, 0, 0], dtype=np.uint8)
    green = np.array([0, 128, 0], dtype=np.uint8)
    blue = np.array([0, 0, 128], dtype=np.uint8)
    purple = np.array([128, 0, 128], dtype=np.uint8)
    yellow = np.array([128, 128, 0], dtype=np.uint8)
    teal = np.array([0, 128, 128], dtype=np.uint8)


def resize(imgNDArray, scale_factor) -> np.ndarray:
    """Resize image by scale factor using cv2 interarea"""
    return cv2.resize(
        img,
        (img.shape[0] * scale_factor, img.shape[1] * scale_factor),
        interpolation=cv2.INTER_AREA,
    )


def draw_text_at(
    imgNDArray,
    text: str,
    text_anchor: Tuple[int, int],
    text_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Use cv2 to draw anti-aliased text with default font text

    Args:
        img: RGB Image
        text: Text to render
        text_anchor: Text anchor in terms of (w,h) coordinates from top left
        text_color: RGB tuple
    """
    cv2.putText(
        img,
        text,
        text_anchor,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.25,
        text_color,
        1,
        lineType=cv2.LINE_AA,
    )
    return img


def tile_images(img_nhwc: Sequence[np.ndarray]) -> np.ndarray:  # pragma: no cover
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    :param img_nhwc: list or array of images, ndim=4 once turned into array. img nhwc
        n = batch index, h = height, w = width, c = channel
    :return: img_HWc, ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    n_images, height, width, n_channels = img_nhwc.shape
    # new_height was named H before
    new_height = int(np.ceil(np.sqrt(n_images)))
    # new_width was named W before
    new_width = int(np.ceil(float(n_images) / new_height))
    img_nhwc = np.array(
        list(img_nhwc)
        + [img_nhwc[0] * 0 for _ in range(n_images, new_height * new_width)]
    )
    # img_HWhwc
    out_image = img_nhwc.reshape((new_height, new_width, height, width, n_channels))
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape((new_height * height, new_width * width, n_channels))
    return out_image
