import torch

def weighted_bce(y_pred, y_true, weight1=200, weight0=1) :
    #loss = torch.nn.BCELoss()
    #calc_weight:
    ones = torch.count_nonzero(y_true)
    zeros = y_true.numel()
    if ones != 0:
        weight1 = zeros/ones
    else:
        weight1 = 1000
    loss = torch.nn.BCEWithLogitsLoss()
    weights = (1.0 - y_true) * weight0 + y_true * weight1
    bce = loss(y_true, y_pred)
    #print(bce)
    w_bce = torch.mean(weights * bce)
    #print(w_bce)
    return w_bce


def IoU(y_pred, y_true, smooth=1):
    intersection = torch.sum(torch.abs(y_true * y_pred))
    union = torch.sum(y_true)+torch.sum(y_pred) - intersection
    iou = torch.mean((intersection + smooth) / (union + smooth))
    return iou

