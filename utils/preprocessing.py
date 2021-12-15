import tensorflow as tf


MIN = (-47.825783, -54.434864)
MAX_MIN = (65.8807, 65.837435)
NORM_MEAN = (0.5963917, 0.60218525)
NORM_STD = (0.0397749, 0.040504467)


def normalize(dataset: tf.data.Dataset):
    def chw_to_hwc(inputs, output):
        return tf.transpose(inputs, [1, 2, 0]), tf.transpose(output, [1, 2, 0])

    dataset_hwc = dataset.map(chw_to_hwc)

    offset = tf.constant(MIN * 2, dtype=tf.float32)

    scaling = tf.constant(MAX_MIN * 2, dtype=tf.float32)

    def normalize_fn(inputs, output):
        tmp = tf.math.subtract(inputs, offset)
        return tf.math.divide(tmp, scaling), output

    normalized_dataset = dataset_hwc.map(normalize_fn)

    def hwc_to_chw(inputs, output):
        return tf.transpose(inputs, [2, 0, 1]), tf.transpose(output, [2, 0, 1])

    return normalized_dataset.map(hwc_to_chw)


def standardize(dataset: tf.data.Dataset):
    def chw_to_hwc(inputs, output):
        return tf.transpose(inputs, [1, 2, 0]), tf.transpose(output, [1, 2, 0])

    dataset_hwc = dataset.map(chw_to_hwc)

    offset = tf.constant(MIN * 2, dtype=tf.float32)

    scaling = tf.constant(MAX_MIN * 2, dtype=tf.float32)

    def normalize_fn(inputs, output):
        tmp = tf.math.subtract(inputs, offset)
        return tf.math.divide(tmp, scaling), output

    normalized_dataset = dataset_hwc.map(normalize_fn)

    mean = tf.constant(NORM_MEAN * 2, dtype=tf.float32)
    std = tf.constant(NORM_STD * 2, dtype=tf.float32)

    def standardize_fn(inputs, output):
        tmp = tf.math.subtract(inputs, mean)
        return tf.math.divide(tmp, std), output

    standardized_dataset = normalized_dataset.map(standardize_fn)

    def hwc_to_chw(inputs, output):
        return tf.transpose(inputs, [2, 0, 1]), tf.transpose(output, [2, 0, 1])

    return standardized_dataset.map(hwc_to_chw)
