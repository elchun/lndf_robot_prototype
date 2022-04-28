import torch
from torch.nn import functional as F


def occupancy(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['occ']
    label = (label + 1) / 2.

    loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'] + 1e-5) + (1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)).mean()
    return loss_dict


def occupancy_net(model_outputs, ground_truth, val=False):
    # Good if using sigmoid on output of decoder
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    # print('model outputs: ', model_outputs)
    loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'] + 1e-5) + (1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)).mean()
    return loss_dict


def conv_occupancy_net(model_outputs, ground_truth, val=False):
    # Good if not using sigmoid on output of decoder
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    loss = F.binary_cross_entropy_with_logits(model_outputs['occ'], label)
    loss_dict['occ'] = loss
    return loss_dict


def distance_net(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()

    dist = torch.abs(model_outputs['occ'] - label * 100).mean()
    loss_dict['dist'] = dist

    return loss_dict


def semantic(model_outputs, ground_truth, val=False):
    loss_dict = {}

    label = ground_truth['occ']
    label = ((label + 1) / 2.).squeeze()

    if val:
        loss_dict['occ'] = torch.zeros(1)
    else:
        loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'].squeeze() + 1e-5) + (1 - label) * torch.log(1 - model_outputs['occ'].squeeze() + 1e-5)).mean()

    return loss_dict

def rotated(model_outputs, ground_truth, val=False):
    """
    https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec
    model_outputs = {'standard': <>, 'rot': <>}
    """

    # print('Im running rotated loss')
    # Good if using sigmoid on output of decoder
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    standard_outputs = model_outputs['standard']
    rot_outputs = model_outputs['rot']

    # print('model outputs: ', model_outputs)
    standard_loss_occ = -1 * (label * torch.log(standard_outputs['occ'] + 1e-5) 
        + (1 - label) * torch.log(1 - standard_outputs['occ'] + 1e-5)).mean()
    rot_loss_occ = -1 * (label * torch.log(rot_outputs['occ'] + 1e-5) 
        + (1 - label) * torch.log(1 - rot_outputs['occ'] + 1e-5)).mean()

    loss_dict['occ'] = standard_loss_occ
    return loss_dict


    pass
