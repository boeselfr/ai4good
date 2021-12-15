import numpy as np
import tensorflow as tf


class DatasetStatsEvaluator:
    def __init__(self, dataset):

        if dataset is None:
            raise ValueError("Dataset must be a valid tf.data.Datset object.")
        self.dataset = dataset

        self._examples_count = dataset.reduce(np.int64(0), lambda x, _: x + 1)
        height, width, self.nchan = next(iter(dataset.take(1))).numpy().shape
        num_pixels_per_image = height * width
        self.total_pixels_count = self._examples_count * num_pixels_per_image

        self._pixel_mean = None
        self._pixel_variance = None
        self._pixel_min = None
        self._pixel_max = None

    @property
    def pixel_mean(self):
        if self._pixel_mean is not None:
            return self._pixel_mean

        norm_factor = tf.cast(self.total_pixels_count, tf.float32)

        def reduce_fn(curr, x):
            sum = tf.math.reduce_sum(x, axis=2)
            return curr + tf.math.divide(sum, norm_factor)

        self._pixel_mean = self.dataset.reduce(
            np.zeros(self.nchan, dtype=tf.float32), reduce_fn
        )
        return self._pixel_mean

    @property
    def pixel_variance(self):
        if self._pixel_variance is not None:
            return self._pixel_variance

        norm_factor = tf.cast(self.total_pixels_count - 1, tf.float32)

        def reduce_fn(curr, x):
            ssd = tf.math.reduce_sum(
                tf.math.squared_difference(x, self.pixel_mean), axis=2
            )
            return curr + tf.math.divide(ssd, norm_factor)

        self._pixel_variance = self.dataset.reduce(
            np.zeros(self.nchan, dtype=tf.float32), reduce_fn
        )
        return self._pixel_variance

    @property
    def pixel_min(self):
        if self._pixel_min is not None:
            return self._pixel_min
        self._pixel_min = self.dataset.reduce(
            np.array([np.inf] * self.nchan),
            lambda m, x: tf.math.minimum(m, tf.math.reduce_min(x)),
        )
        return self._pixel_min

    @property
    def pixel_max(self):
        if self._pixel_max is not None:
            return self._pixel_max
        self._pixel_max = self.dataset.reduce(
            np.array([-np.inf] * self.nchan),
            lambda m, x: tf.math.maximum(m, tf.math.reduce_max(x)),
        )
        return self._pixel_max
