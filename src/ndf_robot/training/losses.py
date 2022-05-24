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

def rotated(model_outputs, ground_truth, val=False, **kwargs):
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
    standard_act_hat = torch.flatten(model_outputs['standard_act_hat'], start_dim=1)
    rot_act_hat = torch.flatten(model_outputs['rot_act_hat'], start_dim=1)


    # Calculate loss of occupancy
    standard_loss_occ = -1 * (label * torch.log(standard_outputs['occ'] + 1e-5) 
        + (1 - label) * torch.log(1 - standard_outputs['occ'] + 1e-5)).mean()
    rot_loss_occ = -1 * (label * torch.log(rot_outputs['occ'] + 1e-5) 
        + (1 - label) * torch.log(1 - rot_outputs['occ'] + 1e-5)).mean()
    
    occ_loss = (standard_loss_occ + rot_loss_occ) / 2
    
    # Calculate loss from similarity between latent descriptors 
    device = standard_act_hat.get_device()
    latent_loss = F.cosine_embedding_loss(standard_act_hat, rot_act_hat, 
        torch.ones(standard_act_hat.shape[0]).to(device))

    # latent_loss = F.mse_loss(standard_latent, rot_latent)

    latent_loss = latent_loss.mean()

    latent_loss_scale = 1

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


def rotated_triplet(model_outputs, ground_truth, val=False, **kwargs):
    """
    https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec
    model_outputs = {'standard': <>, 'rot': <>, 
        'standard_latent': <>, 'rot_latent': <>, 'rot_negative_latent': <>}
    """

    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    # Get outputs from dict
    standard_outputs = model_outputs['standard']
    rot_outputs = model_outputs['rot']
    standard_act_hat = torch.flatten(model_outputs['standard_act_hat'], start_dim=1)
    rot_act_hat = torch.flatten(model_outputs['rot_act_hat'], start_dim=1)

    rot_negative_latent = torch.flatten(model_outputs['rot_negative_latent'], 
        start_dim=1)


    # Calculate loss of occupancy
    standard_loss_occ = -1 * (label * torch.log(standard_outputs['occ'] + 1e-5) 
        + (1 - label) * torch.log(1 - standard_outputs['occ'] + 1e-5)).mean()
    rot_loss_occ = -1 * (label * torch.log(rot_outputs['occ'] + 1e-5) 
        + (1 - label) * torch.log(1 - rot_outputs['occ'] + 1e-5)).mean()
    
    occ_loss = (standard_loss_occ + rot_loss_occ) / 2
    
    # Calculate loss from similarity between latent descriptors 
    # device = standard_latent.get_device()
    # latent_loss = F.cosine_embedding_loss(standard_latent, rot_latent, 
    #     torch.ones(standard_latent.shape[0]).to(device))
    
    latent_loss = F.triplet_margin_loss(standard_act_hat, rot_act_hat, 
        rot_negative_latent, margin=0.0001)

    # latent_loss = F.mse_loss(standard_latent, rot_latent)

    latent_loss = latent_loss.mean()

    latent_loss_scale = 1

    loss_dict['occ'] = occ_loss + latent_loss_scale * latent_loss

    print('occ loss: ', occ_loss)
    print('latent loss: ', latent_loss)

    return loss_dict
# Add rotated with contrastive
# contrast to point that was not moved


def rotated_adaptive(model_outputs, ground_truth, it=-1,val=False):
    """
    https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec
    model_outputs = {'standard': <>, 'rot': <>, 
        'standard_latent': <>, 'rot_latent': <>,
        'it': <>}
    """

    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    # Get outputs from dict
    standard_outputs = model_outputs['standard']
    rot_outputs = model_outputs['rot']
    standard_act_hat = torch.flatten(model_outputs['standard_act_hat'], start_dim=1)
    rot_act_hat = torch.flatten(model_outputs['rot_act_hat'], start_dim=1)


    # Calculate loss of occupancy
    standard_loss_occ = -1 * (label * torch.log(standard_outputs['occ'] + 1e-5) 
        + (1 - label) * torch.log(1 - standard_outputs['occ'] + 1e-5)).mean()
    rot_loss_occ = -1 * (label * torch.log(rot_outputs['occ'] + 1e-5) 
        + (1 - label) * torch.log(1 - rot_outputs['occ'] + 1e-5)).mean()
    
    occ_loss = (standard_loss_occ + rot_loss_occ) / 2
    
    # Calculate loss from similarity between latent descriptors 
    device = standard_act_hat.get_device()
    latent_loss = F.cosine_embedding_loss(standard_act_hat, rot_act_hat, 
        torch.ones(standard_act_hat.shape[0]).to(device))

    # latent_loss = F.mse_loss(standard_latent, rot_latent)

    # Can also do adaptive based on the occ loss value
    latent_loss = latent_loss.mean()

    if (it == -1):
        latent_loss_scale = 100
    elif (it < 1000):
        latent_loss_scale = 1
    elif (it < 5000):
        latent_loss_scale = 10
    elif (it < 10000):
        latent_loss_scale = 100
    elif (it < 20000):
        latent_loss_scale = 500
    else:
        latent_loss_scale = 500

    loss_dict['occ'] = occ_loss + latent_loss_scale * latent_loss

    print('scale: ', latent_loss_scale)
    print('occ loss: ', occ_loss)
    print('latent loss: ', latent_loss)

    # occ was 0.14 with scale at 1
    # latent was at 0.006 with scale at 1

    # occ was 0.3 with scale 10000
    # latent was 1.5 * 10^-7 with scale 10000

    # occ was 0.3 with scale 100
    # latent was 1.5 * 10^-5 with scale 100
    return loss_dict

def custom_rotated_triplet(model_outputs, ground_truth, it=-1, val=False, **kwargs):
    """
    https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec
    model_outputs = {'standard': <>, 'rot': <>, 
        'standard_latent': <>, 'rot_latent': <>, 'rot_negative_latent': <>}
    """

    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    # Get outputs from dict
    standard_outputs = model_outputs['standard']
    rot_outputs = model_outputs['rot']
    standard_act_hat = torch.flatten(model_outputs['standard_act_hat'], start_dim=1)
    rot_act_hat = torch.flatten(model_outputs['rot_act_hat'], start_dim=1)

    rot_negative_act_hat = torch.flatten(model_outputs['rot_negative_act_hat'], 
        start_dim=1)

    # print(standard_act_hat[0, :5])
    # print(rot_negative_act_hat[0, :5])


    # Calculate loss of occupancy
    standard_loss_occ = -1 * (label * torch.log(standard_outputs['occ'] + 1e-5) 
        + (1 - label) * torch.log(1 - standard_outputs['occ'] + 1e-5)).mean()
    rot_loss_occ = -1 * (label * torch.log(rot_outputs['occ'] + 1e-5) 
        + (1 - label) * torch.log(1 - rot_outputs['occ'] + 1e-5)).mean()
    
    occ_loss = (standard_loss_occ + rot_loss_occ) / 2
    
    # Calculate loss from similarity between latent descriptors 
    device = standard_act_hat.get_device()
    latent_positive_loss = F.cosine_embedding_loss(standard_act_hat, rot_act_hat, 
        torch.ones(standard_act_hat.shape[0]).to(device))

    latent_negative_loss = F.cosine_embedding_loss(standard_act_hat, rot_negative_act_hat, 
        -torch.ones(standard_act_hat.shape[0]).to(device), margin=0)


    latent_loss_scale_default = 100 
    # if it == -1:
    #     latent_loss_scale = latent_loss_scale_default 
    # elif it < 10000:
    #     latent_loss_scale = 0
    # else:
    #     latent_loss_scale = latent_loss_scale_default 

    # occ_loss_threshold = 0.20
    # if occ_loss < occ_loss_threshold:
    #     latent_loss_scale = latent_loss_scale_default
    # else:
    #     latent_loss_scale = 0

    latent_positive_loss = latent_positive_loss.mean()
    latent_negative_loss = latent_negative_loss.mean()


    # loss_dict['occ'] = occ_loss \
    #     + latent_loss_scale * (latent_positive_loss + latent_negative_loss)

    # Margin determines how accurate the occ reconstruction is
    occ_margin = 0.15
    loss_dict['occ'] = max(occ_loss - occ_margin, 0) \
        + latent_positive_loss + latent_negative_loss

    print('occ loss: ', occ_loss)
    # print('latent_scale: ', latent_loss_scale)
    print('latent pos loss: ', latent_positive_loss)
    print('latent neg loss: ', latent_negative_loss)

    return loss_dict