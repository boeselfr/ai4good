# reads in a tif file as a dataset
# runs a model on the dataset
import os

#import tensorflow.keras
import torch
from torchvision import transforms
from geotiff_crop_dataset import CropDatasetReader
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch import nn
from tqdm import tqdm
import datetime

#import NesNet
from utils.metrics import weighted_bce,IoU
#from NesNet import *
import wandb



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


def train(model, train_loader, val_loader, learning_rate, epochs, device,save_path, pos_weight=200):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ######training#######
    for epoch in range(epochs):
        print('Epoch: [{}/{}]'.format(epoch + 1, epochs))

        # train set
        pbar = tqdm(train_loader)
        model.train()
        skipcount = 0
        train_iou = 0
        train_loss = 0
        train_batch_ct = 0
        for batch in pbar:
            # load image and mask into device memory based on two images in one tif now
            batch = np.nan_to_num(batch)
            image = torch.from_numpy(batch[:,0:2,:,:]).to(device)
            reference_mask = torch.from_numpy(batch[:,4,:,:]).to(device).view(-1, 1, image_size, image_size)
            image = torch.cat((image,reference_mask), 1).to(device)
            mask = torch.from_numpy(batch[:,3,:,:]).to(device).view(-1,1,image_size,image_size)
            ones = torch.count_nonzero(mask)
            if ones == 0:
                skipcount += 1
                continue
            """print(f'zeros: {zeros}')
            print(f'ones: {ones}')
            print(f'pos_weight: {weight}')"""
            train_batch_ct += 1
            # pass images into model
            pred = model(image)

            #print(f'pred shape: {np.shape(pred)})
            #print(f'mask shape: {np.shape(mask)}')
            # get loss
            #loss = weighted_bce(pred, mask, weight1=pos_weight, weight0=1)
            tversky = smp.losses.TverskyLoss(mode='binary', smooth=0.1,alpha = 0.01,  beta = 0.99)
            loss = tversky.compute_score(pred, mask)

            # compute and display progress
            jaccard = smp.losses.JaccardLoss(mode='binary', smooth=0.1)
            mIoU = jaccard(y_pred=pred, y_true=mask)
            loss = 1 - mIoU
            train_iou += mIoU
            train_loss += loss
            # update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            pbar.set_description('Loss: {0:1.4f} | mIoU {1:1.4f}'.format(loss.item(), mIoU))

        train_iou = train_iou / train_batch_ct
        train_loss = train_loss / train_batch_ct
        wandb.log({"loss":train_loss,
                   "iou": train_iou})
        print(f'bacthes skipped: {skipcount}')
        print(f'mean train iou: {train_iou}')
        print(f'mean train loss. {train_loss}')
        #######validation######
        pbar = tqdm(val_loader)
        model.eval()
        mean_iou = 0
        batch_ct = 0
        with torch.no_grad():
            for batch in pbar:
                batch_ct += 1
                # load image and mask into device memory
                batch = np.nan_to_num(batch)
                image = torch.from_numpy(batch[:, 0:2, :, :]).to(device)
                reference_mask = torch.from_numpy(batch[:, 4, :, :]).to(device).view(-1, 1, image_size, image_size)
                image = torch.cat((image, reference_mask), 1).to(device)
                mask = torch.from_numpy(batch[:, 3, :, :]).to(device).view(-1, 1, image_size, image_size)

                # pass images into model
                pred = model(image)

                # get loss
                #loss = weighted_bce(pred, mask, weight1=pos_weight, weight0=1)
                tversky = smp.losses.TverskyLoss(mode='binary', alpha=0.01, beta=0.99)
                loss = tversky.compute_score(pred, mask)

                # compute and display progress
                #mIoU = iou_logger(pred,torch.tensor(mask, dtype=torch.int32, device=device))
                jaccard = smp.losses.JaccardLoss(mode='binary', smooth=0.1)
                mIoU = jaccard(y_pred=pred, y_true=mask)
                mean_iou += mIoU
                pbar.set_description('Val Loss: {0:1.4f} | mIoU {1:1.4f}'.format(loss.item(), mIoU))

        mean_iou = mean_iou/ batch_ct
        wandb.log({'val_iou':mean_iou})
        print(f'mean val iou: {mean_iou}')
        #save after each epoch:
        if epoch%5 == 0:
            torch.save(model, save_path)
    return model, mean_iou


def train_nesnet(model, train_loader, val_loader, epochs):

    #convert to keras format:
    #train_loader = DataGenerator(train_loader)
    #val_loader = DataGenerator(val_loader)
    ######training#######
    for epoch in range(epochs):
        print('Epoch: [{}/{}]'.format(epoch + 1, epochs))

        # train set
        pbar = tqdm(train_loader)
        for batch in pbar:
            # load image and mask into device memory based on two images in one tif now
            batch = np.nan_to_num(batch)
            image = batch[:,0:2,:,:]
            image = np.moveaxis(image, 1, -1)
            reference_mask = batch[:,4,:,:]
            #reference_mask = np.moveaxis(reference_mask, 1, -1)
            reference_mask = np.expand_dims(reference_mask, axis=-1)
            image = np.concatenate((image,reference_mask), axis=-1)
            mask = batch[:,3,:,:]
            #mask = np.moveaxis(mask, 1, -1)
            mask = np.expand_dims(mask, axis=-1)

            metrics = model.train_on_batch(x=image, y=mask)

        #######validation######
        pbar = tqdm(val_loader)
        for batch in pbar:
            # load image and mask into device memory
            batch = np.nan_to_num(batch)
            image = batch[:, 0:2, :, :]
            image = np.moveaxis(image, 1, -1)
            reference_mask = batch[:, 4, :, :]
            #reference_mask = np.moveaxis(reference_mask, 1, -1)
            reference_mask = np.expand_dims(reference_mask, axis=1)
            image = np.concatenate((image, reference_mask), axis=-1)
            mask = batch[:, 3, :, :]
            #mask = np.moveaxis(mask, 1, -1)
            mask = np.expand_dims(mask, axis=1)

            metrics = model.test_on_batch(x=image, y=mask, reset_metrics=False)

            # compute and display progress
            pbar.set_description(f'metrics: {metrics}')

    return model


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



def save_model(model, path):
    torch.save(model, path)


def load_model(path):
    model = torch.load(path)
    return model

if __name__ == '__main__':
    model_path = '/Users/fredericboesel/Documents/master/herbst21/AI4Good/ai4good/models/model.pt'
    #wandb.init(project="ai4good", entity="fboesel")
    #data_dir = '/cluster/scratch/fboesel/data/ai4good/ard_gee'
    data_dir = '/Users/fredericboesel/Documents/master/herbst21/AI4Good/data/ard_gee'
    epochs = 10
    image_size = 256
    batch_size = 8
    device = 'cpu'
    learning_rate = 1e-3
    model = smp.Unet(
        encoder_name="resnet34",
        in_channels=3,
        activation='sigmoid',
        classes=1
    )
    modelname = 'resnet34'

    """wandb.config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "modelname": modelname
    }"""

    """input_shape = [256, 256, 3]
    model = NesNet.Nest_Net2(input_shape, deep_supervision=True)
    output_layer = model.get_layer('output_5')
    print("the output shape is:")
    print(output_layer.output_shape)"""


    train_loader, val_loader = get_dataloader(data_dir, image_size, batch_size)
    #visualize_dataset(train_loader)
    time = datetime.datetime.now()
    save_path = f'models/{modelname}_{epochs}_{time}.pt'

    #trained_model,mIoU = train(model, train_loader, val_loader, learning_rate, epochs, device, save_path=save_path)
    #trained_model = train_nesnet(model, train_loader, val_loader, epochs)
    model = torch.load(model_path,  map_location=torch.device('cpu'))
    visualize_predictions(train_loader, model)
    visualize_predictions(val_loader, model)
    #trained_model.save(save_path)
    #save_model(trained_model, save_path)