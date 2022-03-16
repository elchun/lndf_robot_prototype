import torch
from torch.nn import functional as F


def occupancy(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['occ']
    label = (label + 1) / 2.

    loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'] + 1e-5) + (1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)).mean()
    return loss_dict


def occupancy_net(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    # print('model outputs: ', model_outputs)
    # loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'] + 1e-5) + (1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)).mean()
    # print(-1 * (label * torch.log(model_outputs['occ'] + 1e-5) + (1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)))

    loss = F.binary_cross_entropy_with_logits(model_outputs['occ'], label)
    # print(loss)
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
