import functools
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
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


def get_dataset(
    file_pattern: str,
    include_mso: bool = True,
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
    glob = tf.io.gfile.glob(file_pattern)
    dataset = tf.data.TFRecordDataset(glob, compression_type="GZIP")
    dataset = dataset.map(_parse_tfrecord, num_parallel_calls=5)

    if include_mso:
        features = _FEATURES
    else:
        features = SAR_EXPORT_BANDS + RESPONSES
    to_tuple_fn = functools.partial(_to_tuple, features=features)
    dataset = dataset.map(to_tuple_fn, num_parallel_calls=5)
    return dataset
