import os.path as osp
import torch
import numpy as np
import trimesh
import random
import argparse
import copy
from scipy.spatial.transform import Rotation

from ndf_robot.utils import path_util, trimesh_util, util

class DemoLoader():
    def __init__(self, shapenet_id):
        self.shapenet_id = shapenet_id
        self.obj_class = 'mug'
        self.grasp_data = self._load_grasp_demo_from_shapenet_id()

    def _load_grasp_demo_from_shapenet_id(self):
        demo_exp = 'grasp_rim_hang_handle_gaussian_precise_w_shelf'
        demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', self.obj_class, demo_exp)

        grasp_demo_fn = osp.join(demo_load_dir, 'grasp_demo_%s.npz' % self.shapenet_id)
        assert osp.exists(grasp_demo_fn), 'Invalid demo fn!'
        grasp_data = np.load(grasp_demo_fn, allow_pickle=True)
        return grasp_data

    def pose_ee(self, ee_pts):
        ee_pcd = trimesh.PointCloud(ee_pts)

        # Move gripper to appropriate location
        gripper_pose_mat = util.matrix_from_pose(util.list2pose_stamped(self.grasp_data['ee_pose_world']))
        ee_pcd.apply_transform(gripper_pose_mat)
        gripper_pts = np.asarray(ee_pcd.vertices)

        return gripper_pts

    @staticmethod
    def get_gripper_pts(pnt_type='full_hand', translate=True, n_pts=1000):
        """
        Get point cloud of pnt_type

        Args:
            pnt_type (str, optional): ('full_hand', 'bounding_box'). Defaults to 'full_hand'.

        Returns:
            ndarray: gripper points (n x 3)
        """
        n_pts = 1000
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

        # Transform gripper to appropriate location due to pybullet offset
        if translate:
            output_pts_pcd.apply_translation([0, 0, -0.105]) # Shift gripper so jaws align with pose

        output_pts= np.asarray(output_pts_pcd.vertices)
        return output_pts 
        
    def get_object_pts(self):
        """
        Return object point cloud from grasp_data

        Args:
            grasp_data (npz dict): various parameters for grasp data

        Returns:
            ndarray: Point cloud representing object (n x 3)
        """
        data = self.grasp_data
        obj_pts = data['object_pointcloud']  # observed shape point cloud at start
        obj_pts_mean = np.mean(obj_pts, axis=0)
        inliers = np.where(np.linalg.norm(obj_pts - obj_pts_mean, 2, 1) < 0.2)[0]
        obj_pts = obj_pts[inliers]
        obj_pcd = trimesh.PointCloud(obj_pts)

        # Attempt to downsample (may not work rn)
        rix = np.random.permutation(obj_pcd.shape[0])
        obj_pcd_disp = obj_pcd[rix[:int(obj_pcd.shape[0]/5)]]
        obj_pts = np.asarray(obj_pcd.vertices)

        # Shuffle for better sampling
        np.random.shuffle(obj_pts)

        return obj_pts
    
    def visualize_pick_pose(self, gripper_pts, obj_pts):
        gripper_pose_mat = util.matrix_from_pose(util.list2pose_stamped(self.grasp_data['ee_pose_world']))

        # Create scene
        scene = trimesh_util.trimesh_show([gripper_pts, obj_pts], show=False)

        # Draw gripper sphere
        grasp_sph = trimesh.creation.uv_sphere(0.005)
        grasp_sph.apply_transform(gripper_pose_mat)

        # Draw object sphere
        obj_sph = trimesh.creation.uv_sphere(0.005)
        pick_demo_obj_pose = self.grasp_data['obj_pose_world']
        demo_obj_mat = util.matrix_from_pose(util.list2pose_stamped(pick_demo_obj_pose))
        obj_sph.apply_transform(demo_obj_mat)

        scene.add_geometry([grasp_sph, obj_sph])
        scene.show()


    

