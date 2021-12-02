import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset_reader import CropDatasetReader
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tfrecord.torch.dataset import TFRecordDataset, MultiTFRecordDataset


def visualize_dataset(dataloader):
    for i, x in enumerate(dataloader):
            print(np.shape(x))
            x = np.nan_to_num(x)
            image = x[:,0:3,:,:]
            mask = x[:,3,:,:]
            image[:,2,:,:] = image[:,0,:,:]/image[:, 1, :, :]
            print(f'im shape: {np.shape(image)}')
            print(f'mask shape: {np.shape(mask)}')
            # do the rgb rendering in here:
            print(np.shape(image[0]))
            im = np.moveaxis(image[0], 0, -1)
            print(np.shape(im))
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(im) # vmin=[-20,-20,0], vmax=[0,0,2])
            ax[1].imshow(mask[0])
            plt.show()
            break


def get_dataloader(data_dir, image_size, batch_size):
    dataset_list = []

    for i, image in enumerate(os.listdir(data_dir)):
        print(image)
        if 'tif' in image:
            dataset_list.append(CropDatasetReader(
                os.path.join(data_dir, image),
                crop_size=image_size,  # Edge size of each cropped square section
                stride=image_size,  # Number of pixels between each cropped sub-image
                padding=0,  # Number of pixels appended to sides of cropped images
                fill_value=0,  # The value to use for nodata sections and padded regions
                transform=transforms.ToTensor()  # torchvision transform functions
            ))

    # concatenate datasets to one big one:
    ds = torch.utils.data.ConcatDataset(dataset_list)
    # now we can dataload it:
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size

    print(f' trainsize: {train_size}')
    print(f' valsize: {val_size}')

    train_dataset, val_dataset = random_split(ds, [train_size, val_size])

    """# VALIDATE DAT:
    for i in range(3):
        image, mask = train_dataset[i]
        print(f'image shape: {np.shape(image)}')
        print(f'mask shape: {np.shape(mask)}')

        visualize(image=image[0], mask=mask.squeeze(-1))"""

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=False)

    return train_loader, val_loader


def get_dataloader_tfrecords(data_dir, batch_size):
    train_loader = None
    val_loader = None
    return train_loader, val_loader


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def visualize_predictions(dataloader, model):
    # eval visually aswell:
    final_predictions, input_images, masks = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # load image and mask into device memory
            batch = np.nan_to_num(batch)
            rgb_image = torch.from_numpy(batch[:, 0:3, :, :]).to(device)
            image = torch.from_numpy(batch[:, 0:2, :, :])
            reference_mask = torch.from_numpy(batch[:, 4, :, :]).to(device).view(-1, 1, image_size, image_size)
            image = torch.cat((image, reference_mask), 1).to(device)

            mask = torch.from_numpy(batch[:, 3, :, :]).to(device).view(-1, 1, image_size, image_size)

            # pass images into model
            pred = model(image)

            # compute class predictions, i.e. flood or no-flood
            class_pred = torch.round(pred)

            # convert class prediction to numpy
            class_pred = class_pred.detach().cpu().numpy()

            # add to final predictions
            final_predictions.append(class_pred)

            # collect input to model
            #input_images.append(rgb_image.detach().cpu().numpy())
            #masks.append(mask.detach().cpu().numpy())
            rgb_im = rgb_image.detach().cpu().numpy()
            msk = mask.detach().cpu().numpy()

            if np.count_nonzero(class_pred) > 0:
                rgb_im[:, 2, :, :] = 1 - (rgb_im[:, 0, :, :] / rgb_im[:, 1, :, :])
                # getting correct format for plotly -> moveaxis
                rgb_im = np.moveaxis(rgb_im, 1, -1)

                msk = np.concatenate(msk, axis=0)
                msk = np.moveaxis(msk, 1, -1)
                class_pred = np.moveaxis(class_pred, 1, -1)
                for i in range(len(rgb_im)):
                    fig, ax = plt.subplots(1, 3)
                    ax[0].imshow(rgb_im[i])  # vmin=[-20,-20,0], vmax=[0,0,2])
                    ax[1].imshow(msk[i])
                    ax[2].imshow(class_pred[i])
                    plt.show()





    """final_predictions = np.concatenate(final_predictions, axis=0)
    input_images = np.concatenate(input_images, axis=0)
    # adjusting sar to rgb
    input_images[:, 2, :, :] = 1 - (input_images[:, 0, :, :] / input_images[:, 1, :, :])
    # getting correct format for plotly -> moveaxis
    input_images = np.moveaxis(input_images, 1, -1)

    masks = np.concatenate(masks, axis=0)
    masks = np.moveaxis(masks, 1, -1)

    for i, pred in enumerate(final_predictions):
        if i > 10:
            break
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(input_images[i])  # vmin=[-20,-20,0], vmax=[0,0,2])
        ax[1].imshow(masks[i])
        ax[2].imshow(pred)
        plt.show()"""

_SAR_BANDS = ["VV_before", "VH_before", "VV_after", "VH_after"]
_MULTISPECTRAL_OPTICAL_BANDS = ["B4", "B3", "B2"]
_RESPONSES = ["deforestation_mask"]
_BANDS = _SAR_BANDS + _MULTISPECTRAL_OPTICAL_BANDS
_FEATURES = _BANDS + _RESPONSES
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
    stacked = tf.transpose(stacked, [0, 1, 2])
    return stacked[:len(_BANDS), :, :], stacked[len(_BANDS):, :, :]


def get_dataset_tf(tf_record_dir):
    records = [os.path.join(tf_record_dir, file) for file in os.listdir(tf_record_dir)]
    dataset = tf.data.TFRecordDataset(records)
    #dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
    #dataset = dataset.map(to_tuple, num_parallel_calls=5)
    return dataset


def get_dataset(tf_record_dir) :
    #records = [os.path.join(tf_record_dir, file) for file in os.listdir(tf_record_dir)]
    record_pattern = tf_record_dir + '/{}.tfrecord'
    index_path = None
    description = {"VV_before": "float", "VH_before": "float", "VV_after": "float", "VH_after":"float", "B4": "float",
                   "B3": "float", "B2": "float", "deforestation_mask": "float"}
    #dataset = tf.data.TFRecordDataset(records)
    """def decode_image(features):
        # get BGR image from bytes
        features["image"] = cv2.imdecode(features["image"], -1)
        return features"""
    splits = dict()
    record_files = len(os.listdir(tf_record_dir))
    for i,file in enumerate(os.listdir(tf_record_dir)):
        splits[f'{file.split(sep=".")[0]}'] = 1/record_files
    # out of this you can just call dataloader and before that a transform function. juist need to work ...
    dataset = MultiTFRecordDataset(record_pattern, index_path, splits, description)
    print(dataset)
    #dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
    #dataset = dataset.map(to_tuple, num_parallel_calls=5)
    return dataset



if __name__ == '__main__':
    record_path = '/cluster/scratch/fboesel/data/ai4good/test'
    dataset = get_dataset_tf(record_path)
    print(dataset)
    for raw_record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)

    #torch_dataset, numpy_dataset = convert_tf_to_torch(dataset)
