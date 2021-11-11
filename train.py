# reads in a tif file as a dataset
# runs a model on the dataset
import os
import torch
from torchvision import transforms
from geotiff_crop_dataset import CropDatasetReader
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch import nn
from tqdm import tqdm
from utils.metrics import weighted_bce,IoU


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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=False)

    return train_loader, val_loader


def visualize_dataset(dataloader):
    for i, x in enumerate(dataloader):
            print(np.shape(x))
            x = np.nan_to_num(x)
            image = x[:,0:3,:,:]
            mask = x[:,3,:,:]
            image[:,2,:,:] = image[:,0,:,:]/image[:,1,:,:]
            print(f'im shape: {np.shape(image)}')
            print(f'mask shape: {np.shape(mask)}')
            # do the rgb rendering in here:
            print(np.shape(image[0]))
            im = np.moveaxis(image[0], 0, -1)
            print(np.shape(im))
            fig,ax = plt.subplots(1,2)
            ax[0].imshow(im) # vmin=[-20,-20,0], vmax=[0,0,2])
            ax[1].imshow(mask[0])
            plt.show()
            break


def train(model, train_loader, val_loader, learning_rate, epochs, device, pos_weight=200):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ######training#######
    for epoch in range(epochs):
        print('Epoch: [{}/{}]'.format(epoch + 1, epochs))

        # train set
        pbar = tqdm(train_loader)
        model.train()
        for batch in pbar:
            # load image and mask into device memory based on two images in one tif now
            batch = np.nan_to_num(batch)
            image = torch.from_numpy(batch[:,0:2,:,:])
            #image_20 = torch.from_numpy(batch[:,4:6,:,:])
            #image = torch.cat((image_19,image_20), 1).to(device)
            mask = torch.from_numpy(batch[:,3,:,:]).to(device).view(-1,1,image_size,image_size)
            ones = torch.count_nonzero(mask)
            zeros = mask.numel() - ones
            weight = mask.numel() / ones
            """print(f'zeros: {zeros}')
            print(f'ones: {ones}')
            print(f'pos_weight: {weight}')"""

            # pass images into model
            pred = model(image)

            #print(f'pred shape: {np.shape(pred)})
            #print(f'mask shape: {np.shape(mask)}')
            # get loss
            loss = weighted_bce(pred, mask, weight1=pos_weight, weight0=1)

            # update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute and display progress
            mIoU = IoU(pred, mask)

            pbar.set_description('Loss: {0:1.4f} | mIoU {1:1.4f}'.format(loss.item(), mIoU))

        #######validation######
        pbar = tqdm(val_loader)
        model.eval()
        with torch.no_grad():
            for batch in pbar:
                # load image and mask into device memory
                batch = np.nan_to_num(batch)
                image = torch.from_numpy(batch[:, 0:2, :, :])
                #image_20 = torch.from_numpy(batch[:, 4:6, :, :])
                #image = torch.cat((image_19, image_20), 1).to(device)
                mask = torch.from_numpy(batch[:, 3, :, :]).to(device).view(-1, 1, image_size, image_size)

                # pass images into model
                pred = model(image)

                # get loss
                loss = weighted_bce(pred, mask, weight1=pos_weight, weight0=1)

                # compute and display progress
                #mIoU = iou_logger(pred,torch.tensor(mask, dtype=torch.int32, device=device))
                mIoU = IoU(pred, mask)
                pbar.set_description('Val Loss: {0:1.4f} | mIoU {1:1.4f}'.format(loss.item(), mIoU))

    return model, mIoU


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
            #image_20 = torch.from_numpy(batch[:, 4:6, :, :])
            #image = torch.cat((image_19, image_20), 1).to(device)

            mask = torch.from_numpy(batch[:, 3, :, :]).to(device).view(-1, 1, image_size, image_size)

            # pass images into model
            pred = model(image)

            # compute class predictions, i.e. flood or no-flood
            class_pred = pred.argmax(dim=1)

            # convert class prediction to numpy
            class_pred = class_pred.detach().cpu().numpy()

            # add to final predictions
            final_predictions.append(class_pred)

            # collect input to model
            input_images.append(rgb_image.detach().cpu().numpy())
            masks.append(mask.detach().cpu().numpy())

    final_predictions = np.concatenate(final_predictions, axis=0)
    input_images = np.concatenate(input_images, axis=0)
    # adjusting sar to rgb
    input_images[:, 2, :, :] = 1 - (input_images[:, 0, :, :] / input_images[:, 1, :, :])
    # getting correct format for plotly -> moveaxis
    input_images = np.moveaxis(input_images, 1, -1)

    masks = np.concatenate(masks, axis=0)
    masks = np.moveaxis(masks, 1, -1)

    for i, pred in enumerate(final_predictions):
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(input_images[i])  # vmin=[-20,-20,0], vmax=[0,0,2])
        ax[1].imshow(masks[i])
        ax[2].imshow(pred)
        plt.show()


def save_model(model, path):
    torch.save(model, path)


def load_model(path):
    model = torch.load(path)
    return model


if __name__ == '__main__':
    data_dir = '/Users/fredericboesel/Documents/master/herbst21/AI4Good/ai4good/data/test'

    epochs = 10
    image_size = 256
    batch_size = 8
    device = 'cpu'
    learning_rate = 1e-3
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=4,
        classes=1
    )
    modelname = 'unet_res34'

    train_loader, val_loader = get_dataloader(data_dir, image_size, batch_size)
    visualize_dataset(train_loader)

    trained_model,mIoU = train(model, train_loader, val_loader, learning_rate, epochs, device)

    save_path = f'models/{modelname}_{epochs}_{mIoU}.pt'
    save_model(trained_model, save_path)