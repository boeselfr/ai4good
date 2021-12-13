import argparse
import datetime
import os

import segmentation_models_pytorch as smp
import torch
import wandb

from utils import dataloaders


def train(
    model,
    train_loader,
    val_loader,
    learning_rate,
    epochs,
    device,
    save_path,
    criterion,
    record_metrics=False,
):

    if criterion in ["iou", "jaccard"]:
        loss = smp.utils.losses.JaccardLoss()
        lossname = "jaccard_loss"
    elif criterion == "weighted_bce":
        loss = smp.losses.TverskyLoss(
            mode="binary", smooth=0.1, alpha=0.05, beta=0.95)
        lossname = "loss"
    elif criterion in ["dice", "f1"]:
        loss = smp.utils.losses.DiceLoss()
        lossname = "dice_loss"
    else:
        print("loss not defined!")
        return model, 0

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5),
    ]

    optimizer = torch.optim.Adam(
        [dict(params=model.parameters(), lr=learning_rate)])

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
        print("\nEpoch: {}".format(i))
        torch.cuda.empty_cache()
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)

        if record_metrics:
            # predict on random batch train and val and log:
            wandb.log(
                {
                    "val_iou": valid_logs["iou_score"],
                    "val_f1": valid_logs["fscore"],
                    "val_loss": valid_logs[lossname],
                    "train_iou": train_logs["iou_score"],
                    "train_f1": train_logs["fscore"],
                    "train_loss": train_logs[lossname],
                })
            if i % 25 == 0:
                train_x, train_y = next(iter(train_loader))
                train_out = model(torch.tensor(train_x, device=device))
                # transform for plotting:
                train_x = torch.moveaxis(train_x, 1, -1)
                train_y = torch.moveaxis(train_y, 1, -1)
                train_out = torch.moveaxis(train_out, 1,-1)
                val_x, val_y = next(iter(val_loader))
                val_out = model(torch.tensor(val_x, device=device))
                # transform for plotting:
                val_x = torch.moveaxis(val_x, 1, -1)
                val_y = torch.moveaxis(val_y, 1, -1)
                val_out = torch.moveaxis(val_out, 1, -1)
                for idx in range(batch_size):
                    wandb.log({
                        "train_prediction_series":[
                            wandb.Image(train_x.cpu().detach().numpy()[idx, :, :, 0], caption='vv_before'),
                            wandb.Image(train_x.cpu().detach().numpy()[idx, :, :, 2], caption='vv_after'),
                            wandb.Image(train_x.cpu().detach().numpy()[idx, :, :, 1], caption='vh_before'),
                            wandb.Image(train_x.cpu().detach().numpy()[idx, :, :, 3],caption='vh_after'),
                            wandb.Image(train_y.cpu().detach().numpy()[idx, :, :, :], caption='groundtruth'),
                            wandb.Image(train_out.cpu().detach().numpy()[idx, :, :, :], caption='prediction')],
                        "val_prediction_series": [
                            wandb.Image(val_x.cpu().detach().numpy()[idx, :, :, 0], caption='vv_before'),
                            wandb.Image(val_x.cpu().detach().numpy()[idx, :, :, 2], caption='vv_after'),
                            wandb.Image(val_x.cpu().detach().numpy()[idx, :, :, 1], caption='vh_before'),
                            wandb.Image(val_x.cpu().detach().numpy()[idx, :, :, 3], caption='vh_after'),
                            wandb.Image(val_y.cpu().detach().numpy()[idx, :, :, :], caption='groundtruth'),
                            wandb.Image(val_out.cpu().detach().numpy()[idx, :, :, :], caption='prediction')]
                    })

        # do something (save model, change lr, etc.)
        if max_score < valid_logs["iou_score"]:
            max_score = valid_logs["iou_score"]
            torch.save(model, save_path + "_best_model.pth")
            print("Model saved!")

        if i == int(round(epochs / 2)):
            optimizer.param_groups[0]["lr"] = learning_rate * 0.5
            print("Decrease decoder learning rate to 1e-5!")

    return model, max_score


def _parse_args():
    desc = """
    Train a deforestation detection model.
    """
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        required=True,
        help="The path to the directory containing the dataset",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="The path to the directory where training artifacts should be stored.",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=16,
        help="Batch size used in training.",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=1000,
        help="Number of epochs to train for.",
    )

    parser.add_argument(
        "--record-metrics",
        action="store_true",
        help="Record metrics to W&B",
    )

    return parser.parse_args()


if __name__ == "__main__":

    time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    args = _parse_args()

    epochs = args.epochs
    data_dir_path = args.data_dir
    output_dir_path = args.output_dir
    batch_size = args.batch_size

    image_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.0001
    activation = "sigmoid"
    encoder = "efficientnet-b3"
    criterion = "dice"

    train_data_dir = os.path.join(data_dir_path, "train")
    val_data_dir = os.path.join(data_dir_path, "val")

    models_dir = os.path.join(data_dir_path, "models")
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, f"{encoder}_{epochs}_{time}")

    model = smp.Unet(
        encoder_name=encoder,
        in_channels=4,
        activation=activation,
        classes=1,
    )

    if args.record_metrics:
        config = {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "encoder": encoder,
            "data_dir": data_dir_path,
            "criterion": criterion,
            "activation": activation,
        }
        wandb.init(project="ai4good", config=config)

    train_loader = dataloaders.get_tfrecord_dataloader(
        train_data_dir, batch_size=batch_size
    )
    val_loader = dataloaders.get_tfrecord_dataloader(
        val_data_dir, batch_size=batch_size
    )

    trained_model, mIoU = train(
        model,
        train_loader,
        val_loader,
        learning_rate,
        epochs,
        device,
        save_path=save_path,
        criterion=criterion,
        record_metrics=args.record_metrics,
    )
