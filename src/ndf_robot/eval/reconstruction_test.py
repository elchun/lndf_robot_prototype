import os, os.path as osp
from turtle import shape
from typing import no_type_check_decorator
import torch
import numpy as np
import trimesh
import random
import argparse
import copy
from scipy.spatial.transform import Rotation
import plotly.graph_objects as go
from ndf_robot.utils.plotly_save import plot3d

from ndf_robot.utils import path_util
import ndf_robot.model.vnn_occupancy_net.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.eval.ndf_alignment import NDFAlignmentCheck


def make_cam_frame_scene_dict():
    cam_frame_scene_dict = {}
    cam_up_vec = [0, 1, 0]
    plotly_camera = {
        'up': {'x': cam_up_vec[0], 'y': cam_up_vec[1],'z': cam_up_vec[2]},
        'center': {'x': 0, 'y': 0, 'z': 0},
        'eye': {'x': -0.6, 'y': -0.6, 'z': 0.4},
    }

    plotly_scene = {
        'xaxis': 
            {
                'backgroundcolor': 'rgb(255, 255, 255)',
                'gridcolor': 'white',
                'zerolinecolor': 'white',
                'tickcolor': 'rgb(255, 255, 255)',
                'showticklabels': False,
                'showbackground': False,
                'showaxeslabels': False,
                'visible': False,
                'range': [-0.5, 0.5]},
        'yaxis': 
            {
                'backgroundcolor': 'rgb(255, 255, 255)',
                'gridcolor': 'white',
                'zerolinecolor': 'white',
                'tickcolor': 'rgb(255, 255, 255)',
                'showticklabels': False,
                'showbackground': False,
                'showaxeslabels': False,
                'visible': False,
                'range': [-0.5, 0.5]},
        'zaxis': 
            {
                'backgroundcolor': 'rgb(255, 255, 255)',
                'gridcolor': 'white',
                'zerolinecolor': 'white',
                'tickcolor': 'rgb(255, 255, 255)',
                'showticklabels': False,
                'showbackground': False,
                'showaxeslabels': False,
                'visible': False,
                'range': [-0.5, 0.5]},
    }
    cam_frame_scene_dict['camera'] = plotly_camera
    cam_frame_scene_dict['scene'] = plotly_scene

    return cam_frame_scene_dict

def plotly_create_local_frame(transform=None, length=0.03):
    if transform is None:
        transform = np.eye(4)

    x_vec = transform[:-1, 0] * length
    y_vec = transform[:-1, 1] * length
    z_vec = transform[:-1, 2] * length

    origin = transform[:-1, -1]

    lw = 8
    x_data = go.Scatter3d(
        x=[origin[0], x_vec[0] + origin[0]], y=[origin[1], x_vec[1] + origin[1]], z=[origin[2], x_vec[2] + origin[2]],
        line=dict(
            color='red',
            width=lw
        ),
        marker=dict(
            size=0.0
        )
    )
    y_data = go.Scatter3d(
        x=[origin[0], y_vec[0] + origin[0]], y=[origin[1], y_vec[1] + origin[1]], z=[origin[2], y_vec[2] + origin[2]],
        line=dict(
            color='green',
            width=lw
        ),
        marker=dict(
            size=0.0
        )
    )
    z_data = go.Scatter3d(
        x=[origin[0], z_vec[0] + origin[0]], y=[origin[1], z_vec[1] + origin[1]], z=[origin[2], z_vec[2] + origin[2]],
        line=dict(
            color='blue',
            width=lw
        ),
        marker=dict(
            size=0.0
        )
    )
    # fig = go.Figure(data=[x_data, y_data, z_data])
    # fig.show()

    data = [x_data, y_data, z_data]
    return data


if __name__ == '__main__':
    ### ARG PARSING ###
    parser  = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_recon', action='store_true')
    parser.add_argument('--random_rot', action='store_true', help='Apply random rotation to object')
    args = parser.parse_args()


    ### INIT ###
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)


    ### LOAD OBJECTS ###
    # see the demo object descriptions folder for other object models you can try
    obj_model = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/28f1e7bc572a633cb9946438ed40eeb9/models/model_normalized.obj')
    model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_demo_mug_weights.pth')  


    ### INIT OBJECTS ###
    scale1 = 0.25
    scale2 = 0.4
    obj_mesh = trimesh.load(obj_model, process=False)
    obj_mesh.apply_scale(scale1)

    # apply a random initial rotation to the new shape
    if args.random_rot:
        quat = np.random.random(4)
        quat = quat / np.linalg.norm(quat)
        rot = np.eye(4)
        rot[:-1, :-1] = Rotation.from_quat(quat).as_matrix()
        obj_mesh.apply_transform(rot)



    ### GENERATE POINTCLOUD ###
    obj_pcd = obj_mesh.sample(5000)

    # Mean center pointcloud
    obj_pcd = obj_pcd - np.mean(obj_pcd, axis=0)

    ### INIT MODEL ###
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    viz_path = 'visualization'
    if not osp.exists(viz_path):
        os.makedirs(viz_path)


    model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True, sigmoid=True).cuda()
    # model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=False, sigmoid=True).cuda()
    model.load_state_dict(torch.load(model_path))


    ### PROCESS POINTS ###
    n_pts = 1500


    ### PREDICT REFERENCE SHAPE OCC ###
    ref_shape_pcd = torch.from_numpy(obj_pcd[:n_pts]).float().to(device)
    ref_pcd = ref_shape_pcd[None, :, :]

    # Get bounding box
    shape_np = obj_pcd
    assert len(shape_np.shape) == 2, 'expected pcd to be have two dimensions'
    assert shape_np.shape[-1] == 3, 'expected points to be 3d'
    pcd_mean = np.mean(shape_np, axis=0)
    inliers = np.where(np.linalg.norm(shape_np - pcd_mean, 2, 1) < 0.2)[0]
    shape_np = shape_np[inliers]

    shape_pcd = trimesh.PointCloud(shape_np)
    ref_bb = shape_pcd.bounding_box

    # Get eval points
    eval_pts = ref_bb.sample_volume(10000)

    shape_mi = {}
    shape_mi['point_cloud'] = ref_pcd 
    shape_mi['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(device).detach()
    out = model(shape_mi)

    thresh = 0.1
    in_inds = torch.where(out['occ'].squeeze() > thresh)[0].cpu().numpy()
    out_inds = torch.where(out['occ'].squeeze() < thresh)[0].cpu().numpy()

    in_pts = eval_pts[in_inds]
    out_pts = eval_pts[out_inds]


    cam_frame_scene_dict = make_cam_frame_scene_dict()

    viz_fn = osp.join(viz_path, "recon_test.html")
    print(f'Saving visualization to: {viz_fn}')
    plot3d(
        [in_pts, shape_np],
        ['blue', 'black'], 
        viz_fn,
        scene_dict=cam_frame_scene_dict,
        z_plane=False,
        pts_label_list=['in_pts', 'shape_np'])
