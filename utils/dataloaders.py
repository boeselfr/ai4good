from logging import basicConfig
import os

import numpy as np
import torch
from torch.utils import data

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


def get_tfrecord_dataloader(
    data_dir_path: str,
    batch_size: int = 8,
):
    # Load the dataset from local storage or GCS
    file_pattern = os.path.join(data_dir_path, "*tfrecord*")
    dataset = get_dataset(file_pattern, include_mso=False)

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=128, seed=23)

    # Convert to an interable dataset
    iterable_dataset = TFIterableDataset(dataset)

    # Return the dataloader
    return data.DataLoader(iterable_dataset, batch_size=batch_size)
