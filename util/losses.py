import functools
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def init(self):
        super(DiceLoss, self).init()

    def forward(self, pred, target):
       smooth = 1.0

       pflat = torch.flatten(pred.contiguous(), start_dim=1)
       tflat = torch.flatten(target.contiguous(), start_dim=1)

       intersection = torch.sum(pflat * tflat, dim=1)
       A_sum = torch.sum(pflat, dim=1)
       B_sum = torch.sum(tflat, dim=1)

       dice = ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )
       return 1 - (torch.sum(dice) / pflat.size()[0])


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, pred , target, smooth=1):
        pflat = torch.flatten(pred.contiguous(), start_dim=1)
        tflat = torch.flatten(target.contiguous(), start_dim=1)

        intersection = torch.sum(pflat * tflat , dim=1)
        total = torch.sum(pflat, dim=1) + torch.sum(tflat , dim=1)
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)
        IoU = torch.mean(IoU)

        return 1 - IoU 


class DiceBCELoss(nn.Module):
 def __init__(self, weight=0.01, size_average=True):
    super(DiceBCELoss, self).__init__()
    self.dice_loss = DiceLoss()
    self.weight = weight

 def forward(self, pred, targets, smooth=1):
    bce = nn.functional.binary_cross_entropy(pred, targets, reduction='mean')
    dice = self.dice_loss(pred, targets)

    return dice + self.weight * bce


class DiceL1Loss(nn.Module):
 def __init__(self, weight=0.01, size_average=True):
    super(DiceL1Loss, self).__init__()
    self.dice_loss = DiceLoss()
    self.weight = weight

 def forward(self, pred, targets, smooth=1):
    l1 = nn.functional.l1_loss(pred, targets, reduction='mean')
    dice = self.dice_loss(pred, targets)

    return dice + self.weight * l1


class FBetaLoss(nn.Module):
    def __init__(self, beta=1, epsilon=1e-8):
        super(FBetaLoss, self).__init__()

        self.beta = beta
        self.epsilon = epsilon

    def forward(self, pred, target):
       pr = torch.flatten(pred.contiguous(), start_dim=1)
       gt = torch.flatten(target.contiguous(), start_dim=1)

       tp = torch.sum(gt * pr, dim=1)
       fp = torch.sum(pr, dim=1) - tp
       fn = torch.sum(gt, dim=1) - tp

       P = torch.div(tp, torch.add(tp, fp) + self.epsilon)
       R = torch.div(tp, torch.add(tp, fn) + self.epsilon)

       nom = (1 + self.beta ** 2) * torch.mul(P, R)
       denom = torch.add((self.beta ** 2) * P, R)
       denom = torch.add(denom, self.epsilon) # avoiding zero division

       fbeta = torch.div(nom, denom)

       return 1 - torch.mean(fbeta)


class FBetaScore(nn.Module):
    def __init__(self, beta=1, epsilon=1e-8):
        super(FBetaScore, self).__init__()

        self.beta = beta
        self.epsilon = epsilon

    def forward(self, pred, target):
       pred = torch.round(pred)

       pr = torch.flatten(pred.contiguous(), start_dim=1)
       gt = torch.flatten(target.contiguous(), start_dim=1)

       tp = torch.sum(gt * pr, dim=1)
       fp = torch.sum(pr, dim=1) - tp
       fn = torch.sum(gt, dim=1) - tp

       P = torch.div(tp, torch.add(tp, fp) + self.epsilon)
       R = torch.div(tp, torch.add(tp, fn) + self.epsilon)

       nom = (1 + self.beta ** 2) * torch.mul(P, R)
       denom = torch.add((self.beta ** 2) * P, R)
       denom = torch.add(denom, self.epsilon) # avoiding zero division

       fbeta = torch.div(nom, denom)

       return torch.mean(fbeta)


class Activation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        if activation == None or activation == 'identity':
            self.activation = nn.Identity()
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'softmax2d':
            self.activation = functools.partial(torch.softmax, dim=1)
        elif callable(activation):
            self.activation = activation
        else:
            raise ValueError

    def forward(self, x):
        return self.activation(x)
