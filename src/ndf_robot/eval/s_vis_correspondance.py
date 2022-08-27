import os.path as osp

import numpy as np
import torch
import trimesh

from scipy.spatial.transform import Rotation as R

from ndf_robot.utils import path_util




if __name__ == '__main__':

    # see the demo object descriptions folder for other object models you can try
    obj_model = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/28f1e7bc572a633cb9946438ed40eeb9/models/model_normalized.obj')

    model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_latent_1x50/checkpoints/model_epoch_0008_iter_087000.pth')

    scale = 0.25
    mesh = trimesh.load(obj_model, process=False)
    mesh.apply_scale(scale)

    # apply a random initial rotation to the new shape
    quat = np.random.random(4)
    quat = quat / np.linalg.norm(quat)
    rot = np.eye(4)
    rot[:-1, :-1] = Rotation.from_quat(quat).as_matrix()
    mesh2.apply_transform(rot)

    if args.visualize:
        show_mesh1 = mesh1.copy()
        show_mesh2 = mesh2.copy()

        offset = 0.1
        show_mesh1.apply_translation([-1.0 * offset, 0, 0])
        show_mesh2.apply_translation([offset, 0, 0])

        scene = trimesh.Scene()
        scene.add_geometry([show_mesh1, show_mesh2])
        scene.show()

    pcd1 = mesh1.sample(5000)
    pcd2 = mesh2.sample(5000)  # point cloud representing different shape
    # pcd2 = copy.deepcopy(pcd1)  # debug with the exact same point cloud
    # pcd2 = mesh1.sample(5000)  # debug with same shape but different sampled points

    if use_conv:
        # model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=32, model_type='pointnet', return_features=True, sigmoid=False).cuda()
        model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=64, model_type='pointnet', return_features=True, sigmoid=False).cuda()
    else:
        model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True, sigmoid=True).cuda()


    # model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=False, sigmoid=True).cuda()
    model.load_state_dict(torch.load(model_path))


    ndf_alignment = NDFAlignmentCheck(model, pcd1, pcd2, sigma=args.sigma, trimesh_viz=args.visualize)
    ndf_alignment.sample_pts(show_recon=args.show_recon, render_video=args.video)