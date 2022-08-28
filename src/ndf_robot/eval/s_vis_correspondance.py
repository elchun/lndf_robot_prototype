import os.path as osp

import numpy as np
import torch
from torch.nn import functional as F
import trimesh

from scipy.spatial.transform import Rotation as R
import plotly.express as px

from ndf_robot.utils import path_util, util
from ndf_robot.utils.plotly_save import multiplot

import ndf_robot.model.vnn_occupancy_net.vnn_occupancy_net_pointnet_dgcnn \
    as vnn_occupancy_network
import ndf_robot.model.conv_occupancy_net.conv_occupancy_net \
    as conv_occupancy_network


def get_activations(pcd, query, model):
    """
    Get activations of pcd and query points when passed into model.

    Args:
        pcd (np.ndarray): (n, 3)
        query (np.ndarray): (k, 3)

    Returns:
        np.ndarray: (n, z) where z is the length of activations.
    """

    if torch.cuda.is_available():
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    model_input = {}

    query = torch.from_numpy(query).float().to(dev)
    pcd = torch.from_numpy(pcd).float().to(dev)

    model_input['coords'] = query[None, :, :]
    model_input['point_cloud'] = pcd[None, :, :]
    latent = model.extract_latent(model_input)

    act_torch = model.forward_latent(latent, model_input['coords']).detach()
    act = act_torch.squeeze().cpu().numpy()

    return act


if __name__ == '__main__':

    # seed = 0
    seed = 1

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    use_conv = True
    # use_conv = False
    n_samples = 1000

    if torch.cuda.is_available():
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    # see the demo object descriptions folder for other object models you can try
    # obj_model = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/28f1e7bc572a633cb9946438ed40eeb9/models/model_normalized.obj')
    obj_model = osp.join(path_util.get_ndf_obj_descriptions(), 'bottle_centered_obj_normalized/f4851a2835228377e101b7546e3ee8a7/models/model_normalized.obj')

    if use_conv:
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_high_0/checkpoints/model_epoch_0001_iter_093000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_0/checkpoints/model_epoch_0008_iter_508000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_3/checkpoints/model_epoch_0001_iter_063000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/DEBUG_conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_16/checkpoints/model_epoch_0000_iter_055000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/DEBUG_conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_20/checkpoints/model_epoch_0000_iter_017000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/DEBUG_conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_cos_7/checkpoints/model_epoch_0000_iter_018000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_cos_9/checkpoints/model_epoch_0000_iter_003000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_cos_11/checkpoints/model_epoch_0000_iter_001000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_partial_neg_extreme_0/checkpoints/model_epoch_0000_iter_005000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/DEBUG_conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_cos_2/checkpoints/model_epoch_0000_iter_013000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_cos_ok_1/checkpoints/model_epoch_0000_iter_031000.pth')
        model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_cos_ok_3/checkpoints/model_epoch_0000_iter_034000.pth')
    else:
        model_path = osp.join(path_util.get_ndf_model_weights(), 'multi_category_weights.pth')

    scale = 0.25
    mesh1 = trimesh.load(obj_model, process=False)
    mesh1.apply_scale(scale)

    mesh2 = trimesh.load(obj_model, process=False)
    mesh2.apply_scale(scale)

    # extents = mesh1.extents
    # sample_pts = trimesh.sample.volume_rectangular(extents, n_samples, transform=None)
    sample_pts = mesh1.sample(n_samples)
    upright_sample_pts = sample_pts[:, :]
    ref_pt = mesh1.sample(1)

    # Make mesh 1 upright
    rot1 = np.eye(4)
    rot1[:3, :3] = util.make_rotation_matrix('x', np.pi / 2)
    # rot1 = np.eye(4)
    # rot1[:3, :3] = R.random().as_matrix()
    mesh1.apply_transform(rot1)
    ref_pt = util.transform_pcd(ref_pt, rot1)

    rot2 = np.eye(4)
    rot2[:3, :3] = R.random().as_matrix()
    mesh2.apply_transform(rot2)
    sample_pts = util.transform_pcd(sample_pts, rot2)

    pcd1 = mesh1.sample(5000)
    pcd2 = mesh2.sample(5000)  # point cloud representing different shape

    multiplot([pcd1, pcd2, ref_pt, sample_pts], osp.join(path_util.get_ndf_eval(), 'debug_viz', 'debug_correspondance.html'))

    if use_conv:
        model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=128,
            model_type='pointnet', return_features=True, sigmoid=False, acts='last').cuda()
    else:
        model = vnn_occupancy_network.VNNOccNet(latent_dim=256,
            model_type='pointnet', return_features=True, sigmoid=True).cuda()

    model.load_state_dict(torch.load(model_path))

    # -- Get activations -- #

    ref_act = get_activations(pcd1, ref_pt, model)
    acts = get_activations(pcd2, sample_pts, model)

    ref_act = ref_act[None, :]
    ref_act = np.repeat(ref_act, n_samples, axis=0)

    # print(ref_act)
    # cor = F.l1_loss(torch.from_numpy(acts).float().to(dev),
    #     torch.from_numpy(ref_act).float().to(dev),
    #     reduction='none')

    # With cosine similarity, most similar is 1 and least similar is -1
    cor = F.cosine_similarity(torch.from_numpy(acts).float().to(dev),
        torch.from_numpy(ref_act).float().to(dev), dim=1)

    cor = cor.cpu().numpy()

    # cor = cor.sum(axis=1)

    print(cor.shape)

    plot_pts = sample_pts
    color = cor

    # Cap colors so I can actually see differences in the presence of outliers
    max_color = 1000
    outliers = np.where(color > max_color)
    color[outliers] = max_color

    fig = px.scatter_3d(
        x=plot_pts[:, 0], y=plot_pts[:, 1], z=plot_pts[:, 2], color=color)

    fname = osp.join(path_util.get_ndf_eval(), 'debug_viz', 'debug_correspondance_cor.html')

    fig.write_html(fname)