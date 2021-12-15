import argparse
import os

import ee

ee.Initialize()
import numpy as np
import matplotlib.pyplot as plt

from data.dataset import get_dataset
from utils.visualization import sar_to_grayscale, mask_to_grayscale, mso_to_rgb


def inspect_deforestation_detection_dataset(
    data_dir_path: str,
):
    file_pattern = os.path.join(data_dir_path, "*tfrecord*")
    dataset = get_dataset(
        file_pattern=file_pattern,
        include_mso=True,
        transpose=True,
        min_pos_ratio=0.02,
    )

    plt.ion()
    plt.figure(figsize=(20, 10))
    for inputs, outputs in dataset.as_numpy_iterator():

        sar_before_gray = sar_to_grayscale(inputs[:, :, 0:2])
        sar_after_gray = sar_to_grayscale(inputs[:, :, 2:4])
        mso_rgb = mso_to_rgb(inputs[:, :, -3:])
        mask_gray = mask_to_grayscale(outputs)

        plt.subplot(1, 4, 1)
        plt.imshow(sar_before_gray, cmap="gray")
        plt.subplot(1, 4, 2)
        plt.imshow(sar_after_gray, cmap="gray")
        plt.subplot(1, 4, 3)
        plt.imshow(mask_gray)
        plt.subplot(1, 4, 4)
        plt.imshow(mso_rgb)

        plt.show(block=False)

        plt.waitforbuttonpress()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively inspect deforestation detection dataset"
    )

    parser.add_argument(
        "data_dir_path",
        type=str,
        help="Path to the directory containing the TFRecord dataset.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    inspect_deforestation_detection_dataset(
        data_dir_path=args.data_dir_path,
    )
