from logging import basicConfig
import os

import numpy as np
import torch
from torch.utils import data
import tensorflow as tf
import tensorflow.keras.layers as layers
from data.dataset import get_dataset
from scipy.ndimage.filters import uniform_filter

from data.dataset import get_dataset
from utils import preprocessing


class TFIterableDataset(torch.utils.data.IterableDataset):
    """An IterableDataset interface for a tf.data.Dataset

    Usage:
    tf_dataset = get_dataset(...)
    iterable_dataset = TFIterableDataset(tf_dataset)
    data_loader = DataLoader(iterable_dataset)

    Args:
      tf_dataset: A tensorflow dataset
    """

    def __init__(self, tf_dataset):
        super().__init__()
        self.tf_dataset = tf_dataset

    def __iter__(self):
        for inputs, outputs in self.tf_dataset.as_numpy_iterator():
            yield torch.from_numpy(inputs), torch.from_numpy(outputs)


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same randomn changes.
        self.augment_inputs = tf.keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip(
                    "horizontal_and_vertical", seed=seed
                ),
                # layers.experimental.preprocessing.RandomRotation(0.2),
                # stddev has to be mathced here!
                # layers.GaussianNoise(stddev=1)
            ]
        )
        self.augment_labels = tf.keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip(
                    "horizontal_and_vertical", seed=seed
                ),
                # layers.experimental.preprocessing.RandomRotation(0.2)
            ]
        )

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


class Preprocess(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()

    def call(self, inputs, labels):

        inputs = self.despeckle_inputs(inputs)
        labels = self.despeckle_label(labels)
        return inputs, labels

    def despeckle_inputs(self, inputs):
        # band: SAR data to be despeckled (already reshaped into image dimensions)
        # window: descpeckling filter window (tuple)
        # default noise variance = 0.25
        # assumes noise mean = 0
        # to tf tensor and then
        inputs = inputs.numpy()
        print(np.shape(inputs))
        bands = 4
        filtered_input = np.zeros(shape=np.shape(inputs))
        for i in range(bands):
            filtered_band = self.lee_filter(inputs[:, :, i])
            filtered_input[:, :, i] = filtered_band
        return tf.convert_to_tensor(filtered_input)

    def lee_filter(self, band, window=5, var_noise=0.25):
        mean_window = uniform_filter(band, window)
        mean_sqr_window = uniform_filter(band ** 2, window)
        var_window = mean_sqr_window - mean_window ** 2

        weights = var_window / (var_window + var_noise)
        band_filtered = mean_window + weights * (band - mean_window)
        return band_filtered

    def despeckle_label(self, label):
        label = label.numpy()
        print(np.shape(label))
        bands = 1
        filtered_input = np.zeros(shape=np.shape(label))
        for i in range(bands):
            filtered_band = self.lee_filter(label[:, :, i])
            filtered_input[:, :, i] = filtered_band

        return tf.convert_to_tensor(filtered_input)


def get_tfrecord_dataloader(
    data_dir_path: str,
    batch_size: int = 8,
    normalize: bool = False,
    augmentation: bool = False,
    despeckle: bool = False,
):
    # Load the dataset from local storage or GCS
    file_pattern = os.path.join(data_dir_path, "*tfrecord*")
    dataset = get_dataset(file_pattern, include_mso=False)

    if despeckle:
        dataset = dataset.map(Preprocess())

    if normalize:
        # Rescale between [0,1] then subtract mean and divide by std
        dataset = preprocessing.normalize(dataset)

    if augmentation:
        dataset = dataset.map(Augment())

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=128, seed=23)

    # Convert to an interable dataset
    iterable_dataset = TFIterableDataset(dataset)

    # Return the dataloader
    return data.DataLoader(iterable_dataset, batch_size=batch_size)
