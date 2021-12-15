from logging import basicConfig
import os

import numpy as np
import torch
from torch.utils import data
import tensorflow as tf
import tensorflow.keras.layers as layers
from data.dataset import get_dataset


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
        self.augment_inputs = tf.keras.Sequential([
                layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", seed=seed),
                layers.experimental.preprocessing.RandomRotation(0.2),
                #stddev has to be mathced here!
                layers.GaussianNoise(stddev=1)
            ])
        self.augment_labels = tf.keras.Sequential([
                layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", seed=seed),
                layers.experimental.preprocessing.RandomRotation(0.2)
            ])

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


def get_tfrecord_dataloader(
    data_dir_path: str,
    batch_size: int = 8,
    augmentation: bool = False
):
    # Load the dataset from local storage or GCS
    file_pattern = os.path.join(data_dir_path, "*tfrecord*")
    dataset = get_dataset(file_pattern, include_mso=False)

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=128, seed=23)

    if augmentation:
        dataset = dataset.map(Augment())

    # Convert to an interable dataset
    iterable_dataset = TFIterableDataset(dataset)

    # Return the dataloader
    return data.DataLoader(iterable_dataset, batch_size=batch_size)
