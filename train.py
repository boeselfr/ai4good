# reads in a tif file as a dataset
# runs a model on the dataset
import os

#import tensorflow.keras
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch import nn
from tqdm import tqdm
import datetime
import math
#import NesNet
from utils.metrics import weighted_bce,IoU
#from NesNet import *
import wandb
from torchmetrics.functional import accuracy
from sklearn.metrics import f1_score, roc_auc_score
from utils.data_utils import *


def new_train(model, train_loader, val_loader, learning_rate, epochs, device, save_path, criterion):

    if criterion == ['iou', 'jaccard']:
        loss = smp.utils.losses.JaccardLoss()
    elif criterion == 'weighted_bce':
        loss = smp.losses.TverskyLoss(mode='binary', smooth=0.1, alpha=0.05, beta=0.95)
    elif criterion in ['dice', 'f1']:
        loss = smp.utils.losses.DiceLoss()
    else:
        print('loss not defined!')
        return model, 0

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5)
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=learning_rate)
    ])

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    max_score = 0

    for i in range(0, epochs):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)

        wandb.log({'val_iou': valid_logs['iou_score'],
                   "val_f1": valid_logs['fscore'],
                   "val_loss": valid_logs['dice_loss'],
                   "train_iou": train_logs['iou_score'],
                   "train_f1": train_logs['fscore'],
                   "train_loss": train_logs['dice_loss']
                   })

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, save_path + '_best_model.pth')
            print('Model saved!')

        if i == int(round(epochs/2)):
            optimizer.param_groups[0]['lr'] = learning_rate * 0.5
            print('Decrease decoder learning rate to 1e-5!')

    return model, max_score


def train(model, train_loader, val_loader, learning_rate, epochs, device,save_path, criterion):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_mIoU = 0
    ######training#######
    for epoch in range(epochs):
        print('Epoch: [{}/{}]'.format(epoch + 1, epochs))

        # train set
        pbar = tqdm(train_loader)
        model.train()

        skipcount = 0
        train_iou = 0
        hand_train_iou = 0
        train_loss = 0
        train_batch_ct = 0
        acc_score = 0
        train_f1 = 0

        for batch in pbar:
            # load image and mask into device memory based on two images in one tif now
            x,y = handle_batch_prodes(batch, device, image_size)
            # dont consider empty batches
            ones = torch.count_nonzero(mask)
            if ones == 0:
                skipcount += 1
                continue
            train_batch_ct += 1
            # pass images into model
            pred = model(image)
            jaccard = smp.losses.JaccardLoss(mode='binary', smooth=0.1)
            mIoU = 1-jaccard(pred, mask)
            handIoU = IoU(pred, mask)
            # get loss
            #loss = weighted_bce(pred, mask, weight1=pos_weight, weight0=1)
            if criterion == 'logloss_iou':
                loss = -torch.log(handIoU)
            elif criterion == 'linear_iou':
                loss = 1 - handIoU
            elif criterion == 'weighted_bce':
                loss = weighted_bce(pred, mask)
            elif criterion =='tversky':
                tversky = smp.losses.TverskyLoss(mode='binary', smooth=0.1, alpha=0.05, beta=0.95)
                loss = tversky(pred, mask)
            else:
                print('loss not defined!')
                return model, 0

            acc = accuracy(pred, torch.tensor(mask, dtype= torch.int), average='micro')
            # threshold preds for f1_score:
            #thresh = torch.Tensor([0.5], device=device)
            #thresholded_pred = (pred > thresh).float() * 1
            #f1 = f1_score(mask, thresholded_pred)
            dice = smp.losses.DiceLoss(mode='binary', smooth=0.1) # this is 1- dice coeff
            f1 = 1 - dice(pred, mask)
            train_iou += mIoU
            hand_train_iou += handIoU
            train_loss += loss
            acc_score += acc
            train_f1 += f1
            # update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                f'Loss: {loss.item()} | mIoU {mIoU} | handIoU {handIoU} | acc {acc} | f1 {f1}')

        train_iou = train_iou / train_batch_ct
        train_loss = train_loss / train_batch_ct
        hand_train_iou = hand_train_iou / train_batch_ct
        acc_score = acc_score / train_batch_ct
        train_f1 = train_f1 / train_batch_ct
        wandb.log({"loss":train_loss,
                   "iou": train_iou,
                   "handiou": hand_train_iou,
                   "acc": acc_score,
                   "f1":train_f1})

        #######validation######
        pbar = tqdm(val_loader)
        model.eval()
        val_iou = 0
        hand_val_iou = 0
        batch_ct = 0
        val_acc = 0
        val_f1 = 0

        with torch.no_grad():
            for batch in pbar:
                batch_ct += 1
                # load image and mask into device memory
                batch = np.nan_to_num(batch)
                batch[:, 3, :, :] = np.round(batch[:, 3, :, :] / 255.0)
                image = torch.from_numpy(batch[:, 0:2, :, :]).to(device)
                reference_image = torch.from_numpy(batch[:, -2:, :, :]).to(device).view(-1, 2, image_size, image_size)
                image = torch.cat((image, reference_image), 1).to(device)
                mask = torch.from_numpy(batch[:, 3, :, :]).to(device).view(-1, 1, image_size, image_size)

                # pass images into model
                pred = model(image)

                # compute and display progress
                #mIoU = iou_logger(pred,torch.tensor(mask, dtype=torch.int32, device=device))
                jaccard = smp.losses.JaccardLoss(mode='binary', smooth=0.1)
                mIoU = 1 - jaccard(y_pred=pred, y_true=mask)
                handIoU = IoU(pred, mask)

                acc = accuracy(pred, torch.tensor(mask, dtype=torch.int), average='micro')
                dice = smp.losses.DiceLoss(mode='binary')  # this is 1- dice coeff
                f1 = 1 - dice(pred, mask)
                val_iou += mIoU
                hand_val_iou += handIoU
                val_acc += acc
                val_f1 += f1

                pbar.set_description(
                    f' mIoU {mIoU} | handIoU {handIoU} | acc {acc} | f1 {f1}')

        val_iou = val_iou/ batch_ct
        hand_val_iou = torch.tensor([hand_val_iou / batch_ct], device=device)
        val_acc = val_acc / batch_ct
        wandb.log({'val_iou':val_iou,
                   'val_hand_iou': hand_val_iou,
                   "val_acc": val_acc,
                   "val_f1": val_f1})

        #save after each k epochs:
        curr_save_path = save_path + '_' + str(hand_val_iou.item()) + '.pt'
        if epoch%5 == 0 and hand_val_iou.item() > best_mIoU:
            torch.save(model, curr_save_path)
            best_mIoU = hand_val_iou.item()

    return model, hand_val_iou.item()


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


if __name__ == '__main__':
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    epochs = 1000
    image_size = 256
    batch_size = 16
    device = 'cuda'
    learning_rate = 0.0001
    activation = 'sigmoid'
    encoder = 'resnet34'
    # weighted_bce , iou
    criterion = 'dice'

    base_path = '/cluster/scratch/fboesel/data/ai4good'
    # base_path = '/Users/fredericboesel/Documents/master/herbst21/AI4Good/data'
    data_dir = os.path.join(base_path, 'tf_records')
    save_path = os.path.join(base_path, f'models/{encoder}_{epochs}_{time}')

    model = smp.Unet(
        encoder_name=encoder,
        in_channels=4,
        activation=activation,
        classes=1
    )

    config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "encoder": encoder,
        "data_dir": data_dir,
        "criterion": criterion,
        "activation": activation
    }
    wandb.init(project="ai4good", entity="fboesel", config=config)

    train_loader, val_loader = get_dataloader_tfrecords(data_dir, batch_size)

    trained_model, mIoU = new_train(model, train_loader, val_loader, learning_rate, epochs, device, save_path=save_path, criterion=criterion)

