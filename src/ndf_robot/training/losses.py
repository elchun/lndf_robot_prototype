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
    model_outputs = {'standard': <>, 'rot': <>, 
        'standard_latent': <>, 'rot_latent': <>}
    """

    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    # Get outputs from dict
    standard_outputs = model_outputs['standard']
    rot_outputs = model_outputs['rot']
    standard_latent = torch.flatten(model_outputs['standard_latent'], start_dim=1)
    rot_latent = torch.flatten(model_outputs['rot_latent'], start_dim=1)


    # Calculate loss of occupancy
    standard_loss_occ = -1 * (label * torch.log(standard_outputs['occ'] + 1e-5) 
        + (1 - label) * torch.log(1 - standard_outputs['occ'] + 1e-5)).mean()
    rot_loss_occ = -1 * (label * torch.log(rot_outputs['occ'] + 1e-5) 
        + (1 - label) * torch.log(1 - rot_outputs['occ'] + 1e-5)).mean()
    
    occ_loss = (standard_loss_occ + rot_loss_occ) / 2
    
    # Calculate loss from similarity between latent descriptors 
    device = standard_latent.get_device()
    latent_loss = F.cosine_embedding_loss(standard_latent, rot_latent, 
        torch.ones(standard_latent.shape[0]).to(device))

    # latent_loss = F.mse_loss(standard_latent, rot_latent)

    latent_loss = latent_loss.mean()

    latent_loss_scale = 100

    loss_dict['occ'] = occ_loss + latent_loss_scale * latent_loss

    print('occ loss: ', occ_loss)
    print('latent loss: ', latent_loss)

    # occ was 0.14 with scale at 1
    # latent was at 0.006 with scale at 1

    # occ was 0.3 with scale 10000
    # latent was 1.5 * 10^-7 with scale 10000

    # occ was 0.3 with scale 100
    # latent was 1.5 * 10^-5 with scale 100
    return loss_dict


def rotated_triplet(model_outputs, ground_truth, val=False):
    """
    https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec
    model_outputs = {'standard': <>, 'rot': <>, 
        'standard_latent': <>, 'rot_latent': <>}
    """

    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    # Get outputs from dict
    standard_outputs = model_outputs['standard']
    rot_outputs = model_outputs['rot']
    standard_latent = torch.flatten(model_outputs['standard_latent'], start_dim=1)
    rot_latent = torch.flatten(model_outputs['rot_latent'], start_dim=1)


    # Calculate loss of occupancy
    standard_loss_occ = -1 * (label * torch.log(standard_outputs['occ'] + 1e-5) 
        + (1 - label) * torch.log(1 - standard_outputs['occ'] + 1e-5)).mean()
    rot_loss_occ = -1 * (label * torch.log(rot_outputs['occ'] + 1e-5) 
        + (1 - label) * torch.log(1 - rot_outputs['occ'] + 1e-5)).mean()
    
    occ_loss = (standard_loss_occ + rot_loss_occ) / 2
    
    # Calculate loss from similarity between latent descriptors 
    device = standard_latent.get_device()
    latent_loss = F.cosine_embedding_loss(standard_latent, rot_latent, 
        torch.ones(standard_latent.shape[0]).to(device))

    # latent_loss = F.mse_loss(standard_latent, rot_latent)

    latent_loss = latent_loss.mean()

    latent_loss_scale = 100

    loss_dict['occ'] = occ_loss + latent_loss_scale * latent_loss

    print('occ loss: ', occ_loss)
    print('latent loss: ', latent_loss)

    # occ was 0.14 with scale at 1
    # latent was at 0.006 with scale at 1

    # occ was 0.3 with scale 10000
    # latent was 1.5 * 10^-7 with scale 10000

    # occ was 0.3 with scale 100
    # latent was 1.5 * 10^-5 with scale 100
    return loss_dict
# Add rotated with contrastive
# contrast to point that was not moved