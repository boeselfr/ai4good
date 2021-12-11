import numpy as np

_SAR_GRD_MIN = -20
_SAR_GRD_MAX = 0


def mso_to_rgb(mso_image: np.ndarray, max: float = 0.3):
    """
    Convert multispectral optical image to RGB for visualization.
    Note: the function assumes that the input only has three channels
    """
    if mso_image.shape[2] != 3:
        raise ValueError(
            "Multispectral optical should have "
            "only three channels for conversion to RGB"
        )
    # Rescale to [0, 255]
    conversion_factor = 255 / max
    rgb = np.clip(mso_image * conversion_factor, 0, 255)
    # Round to the closest integer and cast to uint8
    return np.rint(rgb).astype(np.uint8)


def mask_to_grayscale(mask: np.ndarray):
    """
    Convert the given segmentation mask to visualizable grayscale image.
    """
    return np.rint(mask * 255).astype(np.uint8)


def sar_to_rgb(sar_image: np.ndarray):
    """
    Renders a 2-channel SAR image as 3-channel RGB
    """
    if sar_image.shape[2] != 2:
        raise ValueError("SAR should have only two channels for conversion to RGB")
    sar_image_clipped = np.clip(sar_image, _SAR_GRD_MIN, _SAR_GRD_MAX)
    rg = (sar_image_clipped - _SAR_GRD_MIN) * (255 / (_SAR_GRD_MAX - _SAR_GRD_MIN))
    # Create b channel by dividing the first channel by the second
    b = np.clip(sar_image[:, :, 0] / sar_image[:, :, 1], 0, 2) * (255 / 2)
    # Concatenate, round and convert to uint8
    rgb = np.concatenate([rg, np.expand_dims(b, axis=2)], axis=2)
    return np.rint(rgb).astype(np.uint8)
