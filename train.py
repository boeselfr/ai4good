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
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)

        if record_metrics:
            wandb.log(
                {
                    "val_iou": valid_logs["iou_score"],
                    "val_f1": valid_logs["fscore"],
                    "val_loss": valid_logs[lossname],
                    "train_iou": train_logs["iou_score"],
                    "train_f1": train_logs["fscore"],
                    "train_loss": train_logs[lossname],
                }
            )

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
    save_path = os.path.join(models_dir, f"models/{encoder}_{epochs}_{time}")

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
