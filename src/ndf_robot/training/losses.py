from sympy import Q
import torch
from torch.nn import functional as F

from ndf_robot.model.conv_occupancy_net.encoder.pointnet import LocalPoolPointnet


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


def triplet(occ_margin=0, positive_loss_scale=0.3, negative_loss_scale=0.3,
    similar_occ_only=False, positive_margin=0.001, negative_margin=0.1):
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
        similar_occ_only (bool, optional): True to only compare activations
            from points where the ground truth occupancy is true. Defaults to
            False.
        positive_margin (float, optional): margin to use in cosine similarity
            for latent_positive_loss. Defaults to 0.001.
        negative_margin (float, optional): margin to use in cosine similarity
            for latent_negative_loss. Defaults to 0.1

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

        standard_act_hat = model_outputs['standard_act_hat']
        rot_act_hat = model_outputs['rot_act_hat']
        rot_negative_act_hat = model_outputs['rot_negative_act_hat']

        # Mask all occ that are not in the shape
        if similar_occ_only:
            non_zero_label = label.unsqueeze(-1)
            standard_act_hat *= non_zero_label
            rot_act_hat *= non_zero_label
            rot_negative_act_hat *= non_zero_label

        standard_act_hat = torch.flatten(standard_act_hat, start_dim=1)
        rot_act_hat = torch.flatten(rot_act_hat, start_dim=1)
        rot_negative_act_hat = torch.flatten(rot_negative_act_hat, start_dim=1)

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
                torch.ones(standard_act_hat.shape[0]).to(device), margin=positive_margin)
            latent_positive_loss = latent_positive_loss.mean()
        else:
            latent_positive_loss = 0

        if negative_loss_scale > 0:
            # Calculate loss from difference between unrelated latent descriptors
            latent_negative_loss = F.cosine_embedding_loss(standard_act_hat, rot_negative_act_hat,
                -torch.ones(standard_act_hat.shape[0]).to(device), margin=negative_margin)
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


def simple_l2(positive_loss_scale: int = 1, negative_loss_scale: int = 1,
    num_negative_samples: int=100):
    def loss_fn(model_outputs, ground_truth, val=False, **kwargs):
        """
        L2 loss enforcing similarity between rotated activations and
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

        similar_occ_only = True

        loss_dict = dict()
        label = ground_truth['occ'].squeeze()
        label = (label + 1) / 2.

        standard_outputs = model_outputs['standard']
        rot_outputs = model_outputs['rot']

        standard_act_hat = model_outputs['standard_act_hat']
        rot_act_hat = model_outputs['rot_act_hat']

        rot_negative_act_hat = model_outputs['rot_negative_act_hat']

        if similar_occ_only:
            non_zero_label = label.unsqueeze(-1)
            # print('nz_label: ', non_zero_label)
            # print('Shape: ', non_zero_label.size())  # [5, 1500, 1]
            # print('Sum:', non_zero_label.sum())
            standard_act_hat *= non_zero_label
            rot_act_hat *= non_zero_label
            rot_negative_act_hat *= non_zero_label

        # print('Standard shape: ', standard_act_hat.size())  # [6, 1500, 32]

        # print('Flattened standard size: ', standard_act_hat.size())  # Was [6, 48000]
        # print('Flattened rot size: ', rot_act_hat.size())  # Was [6, 48000]

        # Calculate loss of occupancy
        standard_loss_occ = -1 * (label * torch.log(standard_outputs['occ'] + 1e-5)
            + (1 - label) * torch.log(1 - standard_outputs['occ'] + 1e-5)).mean()
        rot_loss_occ = -1 * (label * torch.log(rot_outputs['occ'] + 1e-5)
            + (1 - label) * torch.log(1 - rot_outputs['occ'] + 1e-5)).mean()

        occ_loss = (standard_loss_occ + rot_loss_occ) / 2

        # latent_positive_loss = F.mse_loss(standard_act_hat, rot_act_hat, reduction='mean')
        latent_positive_loss = F.l1_loss(standard_act_hat, rot_act_hat, reduction='mean')

        latent_negative_loss = F.l1_loss(rot_act_hat[:num_negative_samples, :],
            rot_negative_act_hat[:num_negative_samples, :], reduction='mean')

        # latent_positive_loss = F.cosine_similarity(standard_act_hat, rot_act_hat, dim=2)
        # print('Loss size: ', latent_positive_loss.size())
        # latent_positive_loss = latent_positive_loss.mean()

        # latent_negative_loss = 1 - F.cosine_similarity(rot_act_hat[:num_negative_samples, :],
        #     rot_negative_act_hat[:num_negative_samples, :], dim=2)
        # latent_negative_loss = latent_negative_loss.mean()

        # latent_negative_loss = F.mse_loss(rot_act_hat[:num_negative_samples, :],
        #     rot_negative_act_hat[:num_negative_samples, :], reduction='mean')

        overall_loss = occ_loss \
            + positive_loss_scale * latent_positive_loss \
            - negative_loss_scale * latent_negative_loss

        loss_dict['occ'] = overall_loss


        print('occ loss: ', occ_loss)
        print('latent pos loss: ', latent_positive_loss)
        print('latent neg loss: ', -latent_negative_loss)
        print('overall loss: ', overall_loss)

        return loss_dict

    return loss_fn


def rotated_triplet_log(model_outputs, ground_truth, **kwargs):
    """
    Joint loss of occupancy and log of similiarty between rotated and unrotated
    coordinates
    Args:
        model_outputs (dict): Dictionary containing 'standard', 'rot',
            'standard_act_hat', 'rot_act_hat'
        ground_truth (dict): Dictionary containing 'occ'
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

    # Calculate loss from similarity between latent descriptors
    negative_margin = 10**-3
    positive_margin = 10**-8
    device = standard_act_hat.get_device()
    latent_positive_loss = F.cosine_embedding_loss(standard_act_hat, rot_act_hat,
        torch.ones(standard_act_hat.shape[0]).to(device), margin=positive_margin)

    latent_negative_loss = F.cosine_embedding_loss(standard_act_hat, rot_negative_act_hat,
        -torch.ones(standard_act_hat.shape[0]).to(device), margin=negative_margin)

    latent_positive_loss = latent_positive_loss.mean()
    latent_negative_loss = latent_negative_loss.mean()

    negative_loss_scale = 0.5
    positive_loss_scale = 0.05


    # loss_dict['occ'] = occ_loss \
    #     + positive_loss_scale * torch.log(positive_pad + latent_positive_loss) \
    #     + negative_loss_scale * torch.log(torch.tensor(negative_pad)
    #         + max(latent_negative_loss - negative_margin, 0))

    # The negative loss should prevent all the activations from becoming similar
    # to each other while the positive loss encourages rotation invariance
    loss_dict['occ'] = occ_loss \
        + positive_loss_scale * torch.log(10**-5 + latent_positive_loss) \
        + negative_loss_scale * latent_negative_loss \

    print('occ loss: ', occ_loss)
    # print('latent_scale: ', latent_loss_scale)
    print('latent pos loss: ', latent_positive_loss)
    print('latent neg loss: ', latent_negative_loss)

    return loss_dict

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
