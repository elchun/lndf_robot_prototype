import os.path as osp
import torch
import numpy as np
import trimesh
import random
import argparse
import copy
from scipy.spatial.transform import Rotation

from ndf_robot.utils import path_util
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.eval.ndf_gripper_alignment import NDFAlignmentCheck
from ndf_robot.eval.ndf_demo_loader import DemoLoader



if __name__ == '__main__':
    parser  = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_recon', action='store_true')
    parser.add_argument('--sigma', type=float, default=0.025)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)

    model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_demo_mug_weights.pth')  

    gripper_pts = DemoLoader.get_gripper_pts(pnt_type='full_hand')
    query_pts = DemoLoader.get_gripper_pts(pnt_type='bounding_box', translate=True)

    ref_shapenet_id = '6aec84952a5ffcf33f60d03e1cb068dc'
    ref_demo = DemoLoader(ref_shapenet_id) 
    ref_ee_pts = ref_demo.pose_ee(query_pts) # Posed query pts
    ref_obj_pts = ref_demo.get_object_pts()

    target_shapenet_id = '5c48d471200d2bf16e8a121e6886e18d'
    target_demo = DemoLoader(target_shapenet_id)
    target_obj_pts = target_demo.get_object_pts()

    if args.visualize:
        print('Visualizing ref demo')
        ref_demo.visualize_pick_pose(ref_ee_pts, ref_obj_pts)

    model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True, sigmoid=True).cuda()
    model.load_state_dict(torch.load(model_path))

    ndf_alignment = NDFAlignmentCheck(model, ref_obj_pts, target_obj_pts, sigma=args.sigma, trimesh_viz=args.visualize, query_points=None)
    # ndf_alignment = NDFAlignmentCheck(model, ref_obj_pts, target_obj_pts, sigma=args.sigma, trimesh_viz=args.visualize, query_points=ref_ee_pts)
    ndf_alignment.sample_pts(show_recon=args.show_recon, render_video=args.video)
