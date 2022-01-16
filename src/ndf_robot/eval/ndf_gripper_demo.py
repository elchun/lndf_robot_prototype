import os.path as osp
import torch
import numpy as np
import trimesh
import random
import argparse
import copy
from scipy.spatial.transform import Rotation

from ndf_robot.utils import path_util, trimesh_util, util
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.eval.ndf_gripper_alignment import NDFAlignmentCheck

def load_grasp_demo_from_shapenet_id(shapenet_id, obj_class='mug'):
    demo_exp = 'grasp_rim_hang_handle_gaussian_precise_w_shelf'
    demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', obj_class, demo_exp)

    grasp_demo_fn = osp.join(demo_load_dir, 'grasp_demo_%s.npz' % shapenet_id)
    assert osp.exists(grasp_demo_fn), 'Invalid demo fn!'
    grasp_data = np.load(grasp_demo_fn, allow_pickle=True)

    return grasp_data


def get_gripper_pts(pnt_type='full_hand'):
    """
    Get point cloud of pnt_type

    Args:
        pnt_type (str, optional): ('full_hand', 'bounding_box'). Defaults to 'full_hand'.

    Returns:
        ndarray: gripper points (n x 3)
    """
    n_pts = 10000
    # For use as query points
    gripper_mesh_file = osp.join(path_util.get_ndf_descriptions(), 'franka_panda/meshes/panda_open_hand_full.obj')

    # Load and sample gripper mesh
    full_gripper_mesh = trimesh.load_mesh(gripper_mesh_file)
    full_gripper_pts = full_gripper_mesh.sample(n_pts)
    full_gripper_pts_uniform = trimesh.sample.volume_mesh(full_gripper_mesh, n_pts)
    full_gripper_pts_pcd = trimesh.PointCloud(full_gripper_pts_uniform)
    
    full_gripper_bb = full_gripper_pts_pcd.bounding_box
    if pnt_type == 'bounding_box':
        output_pts_pcd = trimesh.PointCloud(full_gripper_bb.sample_volume(n_pts))
    elif pnt_type == 'full_hand':
        output_pts_pcd = full_gripper_pts_pcd
    else:
        raise ValueError('Invalid pnt_type')

    # Transform gripper to appropriate location
    output_pts_pcd.apply_translation([0, 0, -0.105]) # Shift gripper so jaws align with pose
    output_pts= np.asarray(output_pts_pcd.vertices)

    return output_pts 


def get_object_pts(grasp_data):
    """
    Return object point cloud from grasp_data

    Args:
        grasp_data (npz dict): various parameters for grasp data

    Returns:
        ndarray: Point cloud representing object (n x 3)
    """
    data = grasp_data
    demo_obj_pts = data['object_pointcloud']  # observed shape point cloud at start
    demo_pts_mean = np.mean(demo_obj_pts, axis=0)
    inliers = np.where(np.linalg.norm(demo_obj_pts - demo_pts_mean, 2, 1) < 0.2)[0]
    demo_obj_pts = demo_obj_pts[inliers]
    demo_obj_pcd = trimesh.PointCloud(demo_obj_pts)

    # Attempt to downsample (may not work rn)
    rix = np.random.permutation(demo_obj_pcd.shape[0])
    demo_obj_pcd_disp = demo_obj_pcd[rix[:int(demo_obj_pcd.shape[0]/5)]]
    demo_obj_pts = np.asarray(demo_obj_pcd.vertices)

    return demo_obj_pts

def pose_gripper(obj_pts, gripper_pts, grasp_data, visualize=False):

    obj_pcd = trimesh.PointCloud(obj_pts)
    gripper_pcd = trimesh.PointCloud(gripper_pts)

    # Move gripper to appropriate location
    gripper_pose_mat = util.matrix_from_pose(util.list2pose_stamped(grasp_data['ee_pose_world']))
    gripper_pcd.apply_transform(gripper_pose_mat)
    gripper_pts = np.asarray(gripper_pcd.vertices)

    # # Zero center points
    # obj_mean = np.mean(obj_pts, axis=0)
    # obj_pts -= obj_mean
    # gripper_pts -= obj_mean

    if visualize:
        scene = trimesh_util.trimesh_show([gripper_pts, obj_pts], show=False)
        grasp_sph = trimesh.creation.uv_sphere(0.005)
        grasp_sph.apply_transform(gripper_pose_mat)

        obj_sph = trimesh.creation.uv_sphere(0.005)
        pick_demo_obj_pose = grasp_data['obj_pose_world']
        demo_obj_mat = util.matrix_from_pose(util.list2pose_stamped(pick_demo_obj_pose))
        obj_sph.apply_transform(demo_obj_mat)

        scene.add_geometry([grasp_sph, obj_sph])
        scene.show()

    return obj_pts, gripper_pts


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


    # OLD
    ##############
    # GET MODELS #
    ##############

    # see the demo object descriptions folder for other object models you can try
    # obj_model1 = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/28f1e7bc572a633cb9946438ed40eeb9/models/model_normalized.obj')
    # obj_model2 = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/586e67c53f181dc22adf8abaa25e0215/models/model_normalized.obj')

    obj_model1 = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/1d18255a04d22794e521eeb8bb14c5b3/models/model_normalized.obj')
    obj_model2 = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/586e67c53f181dc22adf8abaa25e0215/models/model_normalized.obj')

    model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_demo_mug_weights.pth')  

    ##################
    # GET GRASP DATA #
    ##################

    # shapenet_id = '928a383f79698c3fb6d9bc28c8d8a2c4'
    # shapenet_id = '3143a4accdc23349cac584186c95ce9b'
    # shapenet_id = '5c48d471200d2bf16e8a121e6886e18d'

    # Load demo 
    demo_shapenet_id = '6aec84952a5ffcf33f60d03e1cb068dc'

    demo_grasp_data = load_grasp_demo_from_shapenet_id(demo_shapenet_id)
    # gripper_pts = get_gripper_pts(pnt_type='bounding_box')
    gripper_pts = get_gripper_pts(pnt_type='full_hand')
    demo_obj_pts = get_object_pts(demo_grasp_data)

    # Load target
    # Funky target, don't use 
    # target_shapenet_id = '928a383f79698c3fb6d9bc28c8d8a2c4'

    # Does not work well... need to tune
    target_shapenet_id = '5c48d471200d2bf16e8a121e6886e18d'
    # target_shapenet_id = '3143a4accdc23349cac584186c95ce9b'


    target_grasp_data = load_grasp_demo_from_shapenet_id(target_shapenet_id)
    target_obj_pts = get_object_pts(target_grasp_data)

    # Pose gripper and show pose
    visualize = True
    demo_obj_pts, gripper_pts = pose_gripper(demo_obj_pts, gripper_pts, demo_grasp_data, visualize=args.visualize)


    # ###########################
    # # VISUALIZE LOADED POINTS #
    # ###########################

    # scene = trimesh_util.trimesh_show([demo_gripper_pts, demo_obj_pts], show=False)
    # grasp_sph = trimesh.creation.uv_sphere(0.005)
    # demo_ee_mat = util.matrix_from_pose(util.list2pose_stamped(grasp_data['ee_pose_world']))
    # grasp_sph.apply_transform(demo_ee_mat)

    # obj_sph = trimesh.creation.uv_sphere(0.005)
    # pick_demo_obj_pose = grasp_data['obj_pose_world']
    # demo_obj_mat = util.matrix_from_pose(util.list2pose_stamped(pick_demo_obj_pose))
    # obj_sph.apply_transform(demo_obj_mat)

    # scene.add_geometry([grasp_sph, obj_sph])

    # # target_mesh = trimesh.load(obj_model1, process=False)
    # target_pts = demo_obj_pts

    # # target_pts_pcd = trimesh.PointCloud(target_pts)

    # # scene.add_geometry(target_pts_pcd) #TODO Add this geometry to scene
    # # scene.show()

    ####################
    # SCALE AND ROTATE #
    ####################

    # Old?
    scale1 = 0.25
    scale2 = 0.4
    mesh1 = trimesh.load(obj_model1, process=False)
    mesh1.apply_scale(scale1)
    # mesh2 = trimesh.load(obj_model2, process=False) # different instance, different scaling
    # mesh2.apply_scale(scale2)
    mesh2 = trimesh.load(obj_model1, process=False)  # use same object model to debug SE(3) equivariance
    mesh2.apply_scale(scale1)

    # apply a random initial rotation to the new shape
    quat = np.random.random(4)
    quat = quat / np.linalg.norm(quat)
    rot = np.eye(4)
    rot[:-1, :-1] = Rotation.from_quat(quat).as_matrix()
    mesh2.apply_transform(rot)

    #############
    # VISUALIZE #
    #############

    # if args.visualize:
        # show_mesh1 = mesh1.copy()
        # show_mesh2 = mesh2.copy()

        # # So visualization does not have two models on top of eachother
        # offset = 0.1
        # show_mesh1.apply_translation([-1.0 * offset, 0, 0])
        # show_mesh2.apply_translation([offset, 0, 0])

        # scene = trimesh.Scene()
        # scene.add_geometry([show_mesh1, show_mesh2])
        # scene.show()
    
    #####################
    # MAKE POINT CLOUDS #
    #####################

    # Old
    pcd1 = mesh1.sample(5000)
    pcd2 = mesh2.sample(5000)  # point cloud representing different shape
    # pcd2 = copy.deepcopy(pcd1)  # debug with the exact same point cloud
    # pcd2 = mesh1.sample(5000)  # debug with same shape but different sampled points

    ############################
    # CREATE OCCUPANCY NETWORK #
    ############################

    model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True, sigmoid=True).cuda()
    model.load_state_dict(torch.load(model_path))

    ####################
    # RUN OPTIMIZATION #
    ####################


    # Substitute target point cloud (NEW)
    pcd1 = demo_obj_pts 
    pcd2 = target_obj_pts 

    # pcd2 = demo_obj_pts
    # (27080, 3)

    # Shuffle object points (helps with taking sample)
    np.random.shuffle(pcd1)
    np.random.shuffle(pcd2)

    # trimesh_util.trimesh_show([pcd2])

    # Initialize optimizer object
    ndf_alignment = NDFAlignmentCheck(model, pcd1, pcd2, sigma=args.sigma, trimesh_viz=args.visualize, query_points=gripper_pts)
    # ndf_alignment = NDFAlignmentCheck(model, pcd1, pcd2, sigma=args.sigma, trimesh_viz=args.visualize, query_points=None)

    # Run optimization
    ndf_alignment.sample_pts(show_recon=args.show_recon, render_video=args.video)




    #############
    # QUESTIONS #
    #############


    """
    IMPORTANT 
    - How is the scaling done for objects?
        - Will ask
    - Does this approach seem reasonable / could we define the tasks a bit more 
    concretely?
    - How does pose work? (translation + quaternion right?)
        - Offset is: 0.105 in z
    - Where is gripper and mug origin defined for the pose / what is
    the correct origin location for the gripper?
        - pose relative to center of object --> or object coordinate frame?
    - Point cloud in demo:
        - from depth --> yes
        - 

    CURIOUS 
    - Why trimesh?
        - Easiest to use (most common packages)
    - How to pan in pybullet (can only zoom and orbit)
        - set camera in code, set intrinsic and extrinsic matrix and look at saved image
        - don't use visualiztion too much
    - 

    LOGISTICS
    - CSAIL account (I submitted form so should be ok?)
        - email leslie and say i'm doing urop with yilun
    - 

    MY PLAN
    - Robustly load demos into gripper_demo (code cleanup rn)
    - Load other mug into gripper_demo and scale (what is scale?)
    - mod ndf_gripper_alignment.py to take these point clouds
    - get optimizer to encourage no intersection (must look into more)

    Next steps
    - Understand reconstruction
        - either put points in occupancy or reconstruct mug and compute distances
    - understand optimizer

    - Use optimizer to optimize point cloud to avoid mesh
        - set up occupancy network prediction to predict occupancy at location
        - Goal is to merge with large optimizer.  
        - slide or two summarizing what I did, vid of what i tried / whats is working

    TIMELINE
    - Would like to have everything but optimizer working by thurs or fri
    - Would like to have optimizer working by tues 
    (but not sure exactly how it works rn)
    - 




    """