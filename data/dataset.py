from typing import Optional

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
try:
    # NOTE(albanesg): since we are using PyTorch for training, we have to
    # set this to prevent TensorFlow from allocating all the GPU memory
    # just to make a call to the tf.data API.
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

from data.globals import SAR_EXPORT_BANDS, MSO_EXPORT_BANDS, RESPONSES

_BANDS = SAR_EXPORT_BANDS + MSO_EXPORT_BANDS
_FEATURES = _BANDS + RESPONSES
_KERNEL_SIZE = 256


def _parse_tfrecord(example_proto):
    """The parsing function.
    Read a serialized example into the structure defined by FEATURES_DICT.
    Args:
      example_proto: a serialized Example.
    Returns:
      A dictionary of tensors, keyed by feature name.
    """
    kernel_shape = [_KERNEL_SIZE, _KERNEL_SIZE]
    columns = [
        tf.io.FixedLenFeature(shape=kernel_shape, dtype=tf.float32) for k in _FEATURES
    ]
    features_dict = dict(zip(_FEATURES, columns))
    return tf.io.parse_single_example(
        example_proto,
        features_dict,
    )


def _to_tuple(inputs, features, transpose=False):
    """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
    Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
    Args:
      inputs: A dictionary of tensors, keyed by feature name.
      features: The list of features to include - the last one must be the response i.e. the mask
    Returns:
      A tuple of (inputs, outputs).
    """
    inputsList = [inputs.get(key) for key in features]
    stacked = tf.stack(inputsList, axis=0)

    if transpose:
        # Convert from CHW to HWC
        stacked = tf.transpose(stacked, [1, 2, 0])
        return stacked[:, :, : -len(RESPONSES)], stacked[:, :, -len(RESPONSES)]
    else:
        return stacked[: -len(RESPONSES), :, :], stacked[-len(RESPONSES) :, :, :]


def load_and_parse_dataset(
    file_pattern: str,
) -> tf.data.Dataset:
    """Load dataset from a collection of TFRecord files.

    The absolute paths of the TFRecords should match the given file_pattern.

    Args:
      file_pattern: a glob expression to find the TFRecord files
    """
    glob = tf.io.gfile.glob(file_pattern)
    dataset = tf.data.TFRecordDataset(glob, compression_type="GZIP")
    dataset = dataset.map(_parse_tfrecord, num_parallel_calls=5)
    return dataset


def get_dataset(
    file_pattern: str,
    include_mso: bool = False,
    transpose: bool = False,
    min_pos_ratio: Optional[float] = 0.01,
) -> tf.data.Dataset:
    """Function to fetch a TFRecord deforestation detection dataset from GCS.
    Note: to access GCS you need to authenticate or to set the environment
    variable GOOGLE_APPLICATION_CREDENTIALS to the path to your GCS key.
    Args:
      bucket: the name of the bucket on GCS
      file_pattern: a glob pattern to match the files in the bucket
      include_mso: include multispectral optical in the dataset
    Returns:
      A tuple of (inputs, outputs).
    """
    if min_pos_ratio is not None and (min_pos_ratio < 0 or min_pos_ratio > 1.0):
        raise ValueError("min_pos_ratio must be either null or non-negative and < 1.0")

    dataset = load_and_parse_dataset(file_pattern=file_pattern)

    if include_mso:
        features = _FEATURES
    else:
        features = SAR_EXPORT_BANDS + RESPONSES

    if min_pos_ratio is not None:

        def filter_fn(x):
            output = x.get(RESPONSES[0])
            pos = tf.cast(tf.math.count_nonzero(output), tf.float32)
            total = tf.cast(tf.size(output), tf.float32)
            pos_ratio = pos / total
            return pos_ratio > min_pos_ratio

        dataset = dataset.filter(filter_fn)

    # Convert dataset to (inputs, output) tuples
    def to_tuple_fn(inputs):
        return _to_tuple(
            inputs=inputs,
            features=features,
            transpose=transpose,
        )

    dataset = dataset.map(to_tuple_fn, num_parallel_calls=5)

    return dataset
