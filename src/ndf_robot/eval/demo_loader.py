import os, os.path as osp
import numpy as np
import trimesh
import random
import argparse
import copy
from scipy.spatial.transform import Rotation

from airobot import Robot
from airobot import log_info, log_warn, log_debug, log_critical, set_log_level
from airobot.utils import common
from airobot.utils.common import euler2quat

from ndf_robot.utils import path_util, trimesh_util, util

from ndf_robot.utils.eval_gen_utils import (
    process_demo_data_rack, process_demo_data_shelf, process_xq_data, process_xq_rs_data,
)

class NDF_DemoLoader():
    def __init__(self, args, global_dict, cfg):
        self.args = args
        self.global_dict = global_dict
        self.cfg = cfg
        self.load_shelf = True if self.cfg.DEMOS.PLACEMENT_SURFACE == 'shelf' else False

        self.place_demo_filenames = []
        self.grasp_demo_filenames = []
        self._get_filenames() # set place and grasp filenames
        self._trim_demo_filename_lists()

        self.optimizer_gripper_pts = None 
        self.optimizer_gripper_pts_rs = None 
        self.place_optimizer_pts = None 
        self.place_optimizer_pts_rs = None 

        self.demo_target_info_list = []
        self.demo_rack_target_info_list = []
        self.demo_shapenet_ids = []

        self.grasp_data_list = []
        self.place_data_list = []
        self.demo_rel_mat_list = []

        self.sto_grasp_data = None

    
    def _get_filenames(self):
        # get filenames of all the demo files
        demo_filenames = os.listdir(self.global_dict['demo_load_dir'])
        assert len(demo_filenames), 'No demonstrations found in path: %s!' % self.global_dict['demo_load_dir']

        # strip the filenames to properly pair up each demo file
        grasp_demo_filenames_orig = [osp.join(self.global_dict['demo_load_dir'], fn) for fn in demo_filenames if 'grasp_demo' in fn]  # use the grasp names as a reference

        for i, fname in enumerate(grasp_demo_filenames_orig):
            shapenet_id_npz = fname.split('/')[-1].split('grasp_demo_')[-1]
            place_fname = osp.join('/'.join(fname.split('/')[:-1]), 'place_demo_' + shapenet_id_npz)
            if osp.exists(place_fname):
                self.grasp_demo_filenames.append(fname)
                self.place_demo_filenames.append(place_fname)
            else:
                log_warn('Could not find corresponding placement demo: %s, skipping ' % place_fname)
    
    def _trim_demo_filename_lists(self):
        # Trim demo filename lists to number of demos

        grasp_demo_filenames = self.grasp_demo_filenames
        place_demo_filenames = self.place_demo_filenames
        if self.args.n_demos > 0:
            gp_fns = list(zip(grasp_demo_filenames, place_demo_filenames))
            gp_fns = random.sample(gp_fns, self.args.n_demos)
            grasp_demo_filenames, place_demo_filenames = zip(*gp_fns)
            grasp_demo_filenames, place_demo_filenames = list(grasp_demo_filenames), list(place_demo_filenames)
            log_warn('USING ONLY %d DEMONSTRATIONS' % len(grasp_demo_filenames))
            print(grasp_demo_filenames, place_demo_filenames)
        else:
            log_warn('USING ALL %d DEMONSTRATIONS' % len(grasp_demo_filenames))

        grasp_demo_filenames = grasp_demo_filenames[:self.args.num_demo]
        place_demo_filenames = place_demo_filenames[:self.args.num_demo]
    
    def _load_optimizer_pts(self, grasp_demo_fn, place_demo_fn):
        grasp_data = np.load(grasp_demo_fn, allow_pickle=True)
        place_data = np.load(place_demo_fn, allow_pickle=True)
        optimizer_gripper_pts, rack_optimizer_gripper_pts, shelf_optimizer_gripper_pts = process_xq_data(grasp_data, place_data, shelf=self.load_shelf)
        optimizer_gripper_pts_rs, rack_optimizer_gripper_pts_rs, shelf_optimizer_gripper_pts_rs = process_xq_rs_data(grasp_data, place_data, shelf=self.load_shelf)

        if self.cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
            print('Using shelf points')
            place_optimizer_pts = shelf_optimizer_gripper_pts
            place_optimizer_pts_rs = shelf_optimizer_gripper_pts_rs
        else:
            print('Using rack points')
            place_optimizer_pts = rack_optimizer_gripper_pts
            place_optimizer_pts_rs = rack_optimizer_gripper_pts_rs
            # print("Rack real points")
            # trimesh_util.trimesh_show([place_optimizer_pts_rs])
        
        self.optimizer_gripper_pts = optimizer_gripper_pts
        self.optimizer_gripper_pts_rs = optimizer_gripper_pts_rs
        self.place_optimizer_pts = place_optimizer_pts
        self.place_optimizer_pts_rs = place_optimizer_pts_rs
    
    def _load_single_demo(self, grasp_demo_fn, place_demo_fn):
        grasp_data = np.load(grasp_demo_fn, allow_pickle=True)
        place_data = np.load(place_demo_fn, allow_pickle=True)

        self.grasp_data_list.append(grasp_data)
        self.place_data_list.append(place_data)

        start_ee_pose = grasp_data['ee_pose_world'].tolist()
        end_ee_pose = place_data['ee_pose_world'].tolist()
        place_rel_mat = util.get_transform(
            pose_frame_target=util.list2pose_stamped(end_ee_pose),
            pose_frame_source=util.list2pose_stamped(start_ee_pose)
        )

        place_rel_mat = util.matrix_from_pose(place_rel_mat)
        self.demo_rel_mat_list.append(place_rel_mat)

        self.sto_grasp_data = grasp_data

        if self.cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
            target_info, rack_target_info, shapenet_id = process_demo_data_shelf(grasp_data, place_data, cfg=None)
        else:
            target_info, rack_target_info, shapenet_id = process_demo_data_rack(grasp_data, place_data, cfg=None)

        if self.cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
            rack_target_info['demo_query_pts'] = self.place_optimizer_pts

        self.demo_target_info_list.append(target_info)
        self.demo_rack_target_info_list.append(rack_target_info)
        self.demo_shapenet_ids.append(shapenet_id)
    
    def load_demos(self):
        for i, fname in enumerate(self.grasp_demo_filenames):
            print('Loading demo from fname: %s' % fname)

            grasp_demo_fn = self.grasp_demo_filenames[i]
            place_demo_fn = self.place_demo_filenames[i]

            self._load_single_demo(grasp_demo_fn, place_demo_fn)

            if i == 0:
                self._load_optimizer_pts(grasp_demo_fn, place_demo_fn)
    
    def get_optimizer_gripper_pts(self):
        return self.optimizer_gripper_pts

    def get_optimizer_gripper_pts_rs(self):
        return self.optimizer_gripper_pts_rs
        
    def get_place_optimizer_pts(self):
        return self.place_optimizer_pts
    
    def get_place_optimizer_pts_rs(self):
        return self.place_optimizer_pts_rs
    
    def get_demo_target_info_list(self):
        return self.demo_target_info_list
    
    def get_demo_rack_target_info_list(self):
        return self.demo_rack_target_info_list
    
    def get_demo_shapenet_ids(self):
        return self.demo_shapenet_ids

    def get_arbitrary_grasp_data(self):
        return self.sto_grasp_data
    
    @staticmethod
    def get_gripper_pts(n_gripper_pts=500, pnt_type='full_hand'):
        """
        Get point cloud of pnt_type

        Args:
            pnt_type (str, optional): ('full_hand', 'bounding_box'). Defaults to 'full_hand'.

        Returns:
            ndarray: gripper points (n x 3)
        """
        n_pts = n_gripper_pts 
        # For use as query points
        gripper_mesh_file = osp.join(path_util.get_ndf_descriptions(), 'franka_panda/meshes/panda_open_hand_full.obj')

        # Load and sample gripper mesh
        full_gripper_mesh = trimesh.load_mesh(gripper_mesh_file)
        # full_gripper_pts = full_gripper_mesh.sample(n_pts)


        # Doesn't always make enough pts so have to sample more, then slice
        full_gripper_pts_uniform = trimesh.sample.volume_mesh(full_gripper_mesh, n_pts*3)[:n_pts]
        full_gripper_pts_pcd = trimesh.PointCloud(full_gripper_pts_uniform)
    
        full_gripper_bb = full_gripper_pts_pcd.bounding_box
        if pnt_type == 'bounding_box':
            output_pts_pcd = trimesh.PointCloud(full_gripper_bb.sample_volume(n_pts))
        elif pnt_type == 'full_hand':
            output_pts_pcd = full_gripper_pts_pcd
        else:
            raise ValueError('Invalid pnt_type')

        # Transform gripper to appropriate location
        # output_pts_pcd.apply_translation([0, 0, 0.105]) # Shift gripper so jaws align with pose

        # # Move gripper to appropriate location on mug
        # gripper_pose_mat = util.matrix_from_pose(util.list2pose_stamped(grasp_data['ee_pose_world']))
        # output_pts_pcd.apply_transform(gripper_pose_mat)

        output_pts_pcd.apply_translation([0, 0, -0.105]) # Shift gripper so jaws align with pose
        output_pts = np.asarray(output_pts_pcd.vertices)
        return output_pts 
    
    @staticmethod
    def get_grip_area_pts(n_grip_area_pts=500):
        """
        Load point cloud for grip area
        """

        n_pts = n_grip_area_pts 
        # For use as query points
        grasp_area_mesh_fn = osp.join(path_util.get_ndf_descriptions(), 'franka_panda/meshes/grasp_area.obj')

        # Load and sample gripper mesh
        grasp_area_mesh= trimesh.load_mesh(grasp_area_mesh_fn)
        # full_gripper_pts = full_gripper_mesh.sample(n_pts)


        # Doesn't always make enough pts so have to sample more, then slice
        full_grasp_area_pts_uniform = trimesh.sample.volume_mesh(grasp_area_mesh, n_pts*3)[:n_pts]
        full_grasp_area_pts_pcd = trimesh.PointCloud(full_grasp_area_pts_uniform)
    

        # Transform gripper to appropriate location
        # output_pts_pcd.apply_translation([0, 0, 0.105]) # Shift gripper so jaws align with pose

        # # Move gripper to appropriate location on mug
        # gripper_pose_mat = util.matrix_from_pose(util.list2pose_stamped(grasp_data['ee_pose_world']))
        # output_pts_pcd.apply_transform(gripper_pose_mat)

        full_grasp_area_pts_pcd.apply_translation([0, 0, -0.105]) # Shift gripper so jaws align with pose
        output_pts = np.asarray(full_grasp_area_pts_pcd.vertices)
        return output_pts 
