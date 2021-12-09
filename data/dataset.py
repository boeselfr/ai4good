import tensorflow as tf

from data.globals import SAR_EXPORT_BANDS, MSO_EXPORT_BANDS, RESPONSES

_BANDS = SAR_EXPORT_BANDS + MSO_EXPORT_BANDS
_FEATURES = _BANDS + RESPONSES
_KERNEL_SIZE = 256


def parse_tfrecord(example_proto):
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


def to_tuple(inputs):
    """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
    Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
    Args:
      inputs: A dictionary of tensors, keyed by feature name.
    Returns:
      A tuple of (inputs, outputs).
    """
    inputsList = [inputs.get(key) for key in _FEATURES]
    stacked = tf.stack(inputsList, axis=0)
    # Convert from CHW to HWC
    stacked = tf.transpose(stacked, [1, 2, 0])
    return stacked[:, :, : len(_BANDS)], stacked[:, :, len(_BANDS) :]


def get_dataset(
    bucket: str,
    file_pattern: str,
) -> tf.data.Dataset:
    glob = tf.io.gfile.glob("gs://" + bucket + "/" + file_pattern)
    dataset = tf.data.TFRecordDataset(glob, compression_type="GZIP")
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
    dataset = dataset.map(to_tuple, num_parallel_calls=5)
    return dataset
