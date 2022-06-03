from sympy import Q
import torch
from torch.nn import functional as F


def occupancy(model_outputs, ground_truth, val=False):
    """
    LEGACY DO NOT USE???
    """
    loss_dict = dict()
    label = ground_truth['occ']
    label = (label + 1) / 2.

    loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'] + 1e-5)
        + (1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)).mean()
    return loss_dict


def occupancy_net(model_outputs, ground_truth, val=False, **kwargs):
    """
    NLL loss for predicting occupacny

    Args:
        model_outputs (dict): Dictionary with the key 'occ' corresponding to
            tensor
        ground_truth (dict): Dictionary with the key 'occ' corresponding to
            tensor
        val (bool, optional): Unused. Defaults to False.

    Returns:
        dict: Dictionary containing 'occ' which corresponds to tensor of loss
            for each element in batch
    """
    # Good if using sigmoid on output of decoder
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    # print('model outputs: ', model_outputs)
    loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'] + 1e-5)
        + (1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)).mean()
    return loss_dict


def conv_occupancy_net(model_outputs, ground_truth, val=False, **kwargs):
    """
    NLL loss for predicting occupancy with convolutional neural net
    Good if not using a sigmoid output of occupancy network decoder

    Args:
        model_outputs (dict): Dict containing the key 'standard' which maps
            to a dictionary which has the key 'occ' which maps to a
            tensor :(
        ground_truth (dict): Dict with the key 'occ' which maps to a tensor
            of ground truth occupancies
        val (bool, optional): If this is a validation set (Unused).
            Defaults to False.

    Returns:
        dict: Dictionary with key 'occ' corresponding to loss of occupancy
    """
    standard_output = model_outputs['standard']

    # Good if not using sigmoid on output of decoder
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    # print('model outputs: ', model_outputs)
    occ_loss = -1 * (label * torch.log(standard_output['occ'] + 1e-5)
        + (1 - label) * torch.log(1 - standard_output['occ'] + 1e-5)).mean()

    loss_dict['occ'] = occ_loss
    print('occ_loss: ', occ_loss)
    return loss_dict


def distance_net(model_outputs, ground_truth, val=False):
    """
    UNUSED
    """
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()

    dist = torch.abs(model_outputs['occ'] - label * 100).mean()
    loss_dict['dist'] = dist

    return loss_dict


def semantic(model_outputs, ground_truth, val=False):
    """
    UNUSED
    """
    loss_dict = {}

    label = ground_truth['occ']
    label = ((label + 1) / 2.).squeeze()

    if val:
        loss_dict['occ'] = torch.zeros(1)
    else:
        loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'].squeeze() + 1e-5)
            + (1 - label) * torch.log(1 - model_outputs['occ'].squeeze() + 1e-5)).mean()

    return loss_dict


def triplet(occ_margin=0, positive_loss_scale=0.3, negative_loss_scale=0.3):
    """
    Create triplet loss function enforcing similarity between rotated
    activations and difference between random coordinates defined as:

    loss = max(occ_loss - occ_margin, 0) \
        + positive_loss_scale * latent_positive_loss \
        + negative_loss_scale * latent_negative_loss

    where latent_positive_loss is cosine similarity between activations and
    activations of rotated shape while latent_negative_loss is cosine
    difference between rotated latent and a activations of a randomly sampled
    point

    Args:
        occ_margin (float, optional): Loss from occupancy is 0 when below
            margin. Defaults to 0.
        positive_loss_scale (float, optional): Influence of positive loss on
            combined loss. Defaults to 0.3.
        negative_loss_scale (float, optional): Influence of negative loss on
            combined loss. Defaults to 0.3.

    Returns:
        function(model_ouputs, ground_truth): Loss function that takes model
            outputs and ground truth
    """

    def loss_fn(model_outputs, ground_truth, val=False, **kwargs):
        """
        Triplet loss enforcing similarity between rotated activations and
        difference between random coords

        Args:
            model_outputs (dict): Dictionary containing 'standard', 'rot',
                'standard_act_hat', 'rot_act_hat'
            ground_truth (dict): Dictionary containing 'occ'
            occ_margin (float, optional): Lower value makes occupancy
                prediction better

        Returns:
            dict: dict containing 'occ'
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

        # Calculate loss of occupancy
        standard_loss_occ = -1 * (label * torch.log(standard_outputs['occ'] + 1e-5)
            + (1 - label) * torch.log(1 - standard_outputs['occ'] + 1e-5)).mean()
        rot_loss_occ = -1 * (label * torch.log(rot_outputs['occ'] + 1e-5)
            + (1 - label) * torch.log(1 - rot_outputs['occ'] + 1e-5)).mean()

        occ_loss = (standard_loss_occ + rot_loss_occ) / 2

        device = standard_act_hat.get_device()
        if positive_loss_scale > 0:
            # Calculate loss from similarity between latent descriptors
            latent_positive_loss = F.cosine_embedding_loss(standard_act_hat, rot_act_hat,
                torch.ones(standard_act_hat.shape[0]).to(device), margin=0.001)
            latent_positive_loss = latent_positive_loss.mean()
        else:
            latent_positive_loss = 0

        if negative_loss_scale > 0:
            # Calculate loss from difference between unrelated latent descriptors
            latent_negative_loss = F.cosine_embedding_loss(standard_act_hat, rot_negative_act_hat,
                -torch.ones(standard_act_hat.shape[0]).to(device), margin=0.1)
            latent_negative_loss = latent_negative_loss.mean()
        else:
            latent_negative_loss = 0


        loss_dict['occ'] = max(occ_loss - occ_margin, 0) \
            + positive_loss_scale * latent_positive_loss \
            + negative_loss_scale * latent_negative_loss

        print('occ loss: ', occ_loss)
        print('latent pos loss: ', latent_positive_loss)
        print('latent neg loss: ', latent_negative_loss)

        return loss_dict

    return loss_fn


def rotated_log(model_outputs, ground_truth, it=-1):
    """
    Joint loss of occupancy and log of similiarty between rotated and unrotated
    coordinates

    Appears to overfit and reduce magnitude of all activations

    Args:
        model_outputs (dict): Dictionary containing 'standard', 'rot',
            'standard_act_hat', 'rot_act_hat'
        ground_truth (dict): Dictionary containing 'occ'
        it (int, optional): current number of iterations. Defaults to -1.
    """
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    # Get outputs from dict
    standard_outputs = model_outputs['standard']
    rot_outputs = model_outputs['rot']
    standard_act_hat = torch.flatten(model_outputs['standard_act_hat'], start_dim=1)
    rot_act_hat = torch.flatten(model_outputs['rot_act_hat'], start_dim=1)

    # rot_negative_act_hat = torch.flatten(model_outputs['rot_negative_act_hat'],
    #     start_dim=1)

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

    latent_positive_loss = latent_positive_loss.mean()
    positive_loss_scale = 1  # Higher scale makes slope less steep
    positive_loss_log_scale = 0.1
    # margin = 4 * 10 ** -8
    margin = 10 ** -9

    loss_dict['occ'] = occ_loss + positive_loss_log_scale * torch.log(
        margin + positive_loss_scale * latent_positive_loss)

    print('occ loss: ', occ_loss)
    print('latent pos loss: ', latent_positive_loss)

    return loss_dict
