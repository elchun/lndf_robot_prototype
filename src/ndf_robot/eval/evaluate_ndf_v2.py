"""
New version of evaluate_ndf.py

Includes:
- refactored code
- ability to use occupancy
"""
import os, os.path as osp
import random
from socket import TIPC_DEST_DROPPABLE
import numpy as np
import time
import signal
from sympy import evaluate
import torch
import argparse
import shutil

import pybullet as p

from airobot import Robot
from airobot import log_info, log_warn, log_debug, log_critical, set_log_level
from airobot.utils import common
from airobot import log_info
from airobot.utils.common import euler2quat
from ndf_robot.eval.ndf_demo_loader_v2 import DemoLoader

import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.utils import util, trimesh_util
from ndf_robot.utils.util import np2img

from ndf_robot.opt.optimizer_modular import OccNetOptimizer
from ndf_robot.robot.multicam import MultiCams
from ndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from ndf_robot.config.default_obj_cfg import get_obj_cfg_defaults
from ndf_robot.utils import path_util
from ndf_robot.share.globals import bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list
from ndf_robot.utils.franka_ik import FrankaIK
from ndf_robot.utils.eval_gen_utils import (
    soft_grasp_close, constraint_grasp_close, constraint_obj_world, constraint_grasp_open, 
    safeCollisionFilterPair, object_is_still_grasped, get_ee_offset, post_process_grasp_point,
    process_demo_data_rack, process_demo_data_shelf, process_xq_data, process_xq_rs_data, safeRemoveConstraint,
)

# NEW IMPORTS
import trimesh

def get_gripper_pts(grasp_data, n_gripper_pts=500, pnt_type='full_hand'):
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


class Evaluate_NDF():
    def __init__(self, args, global_dict):
        ### SEED ###
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        ### INIT ROBOT ###
        if args.debug:
            set_log_level('debug')
        else:
            set_log_level('info')

        ### SET CLASS VARIABLES ###
        self.args = args
        self.global_dict = global_dict

        self.shapenet_obj_dir = global_dict['shapenet_obj_dir'] 
        self.obj_class = global_dict['object_class'] 
        self.eval_save_dir = global_dict['eval_save_dir']

        self.eval_grasp_imgs_dir = osp.join(eval_save_dir, 'grasp_imgs')
        self.eval_teleport_imgs_dir = osp.join(eval_save_dir, 'teleport_imgs')
        util.safe_makedirs(self.eval_grasp_imgs_dir)
        util.safe_makedirs(self.eval_teleport_imgs_dir)

        self.eval_log_dir = osp.join(eval_save_dir, 'trial_logs')
        util.safe_makedirs(self.eval_log_dir)

        if self.args.use_gripper_occ:
            log_fn = 'log_occ'
        else:
            log_fn = 'log_no_occ'
        self.log_fn = self._get_log_fn(self.eval_log_dir, log_fn)

        self.robot = Robot('franka', pb_cfg={'gui': args.pybullet_viz}, arm_cfg={'self_collision': False, 'seed': args.seed})
        self.ik_helper = FrankaIK(gui=False)

        self.model = None
        self._set_model()

        self.cfg = None
        self._set_cfg()

        self.obj_cfg = None
        self._set_obj_cfg()

        ### SET DEMO LOADER ###
        self.demo_loader = NDF_DemoLoader(self.args, self.global_dict, self.cfg)
        self.demo_loader.load_demos()
        self.demo_shapenet_ids = self.demo_loader.get_demo_shapenet_ids()

        self.gripper_pts = None
        if args.use_gripper_occ:
            print('Using gripper occ')
            self.gripper_pts = self.demo_loader.get_gripper_pts()


        self.test_object_ids = []
        self._set_test_object_ids()

        ### SET OPTIMIZERS ###
        self.grasp_optimizer = OccNetOptimizer(
            self.model,
            query_pts=self.demo_loader.get_optimizer_gripper_pts(),
            query_pts_real_shape=self.demo_loader.get_optimizer_gripper_pts_rs(),
            opt_iterations=self.args.opt_iterations,
            gripper_pts=self.gripper_pts,
            occ_hat_scale=0.5,
        )
        self.grasp_optimizer.set_demo_info(self.demo_loader.get_demo_target_info_list())

        self.place_optimizer = OccNetOptimizer(
            self.model,
            query_pts=self.demo_loader.get_place_optimizer_pts(),
            query_pts_real_shape=self.demo_loader.get_place_optimizer_pts_rs(),
            opt_iterations=self.args.opt_iterations
        )
        self.place_optimizer.set_demo_info(self.demo_loader.get_demo_rack_target_info_list())

        ### SET FINGER IDS ###
        self.finger_joint_id = 9
        self.left_pad_id = 9
        self.right_pad_id = 10

        ### CONFIGURE ROBOT ###
        self._robot_config()

        ### CONFIGURE CAMERAS ###
        self.cams = None
        self.cam_info = {}
        self._set_cams()

        ### SET MORE IDS ###
        self.table_id = None
        self._set_table() # Must occur after cams are set up

        self.rack_link_id = None 
        self._set_rack_link_id()

        self.shelf_link_id = None 
        self._set_shelf_link_id()

        self.placement_link_id = None
        self._set_placement_link_id()

        ### INITIALIZE VARS FOR EVAL ###
        self.viz_data_list = []

        self.success_list = []
        self.place_success_list = []
        self.place_success_teleport_list = []
        self.grasp_success_list = []
    
    def _get_log_fn(self, log_repo_path, base_fn='log_num'):
        f = []
        for (dirpath, dirnames, filenames) in os.walk(log_repo_path):
            f.extend(filenames)
            break
        
        max_num = -1 
        for fn in f:
            name_parts = fn.split('_')
            max_num = max(int(name_parts[-1]), max_num)
        
        return base_fn + '_' + str(max_num + 1)


    def _set_model(self):
        args = self.args
        if args.dgcnn:
            self.model = vnn_occupancy_network.VNNOccNet(
                latent_dim=256, 
                model_type='dgcnn',
                return_features=True, 
                sigmoid=True,
                acts=args.acts).cuda()
        else:
            self.model = vnn_occupancy_network.VNNOccNet(
                latent_dim=256, 
                model_type='pointnet',
                return_features=True, 
                sigmoid=True).cuda()

        if not args.random:
            checkpoint_path = self.global_dict['vnn_checkpoint_path']
            self.model.load_state_dict(torch.load(checkpoint_path))
        else:
            pass
    
    
    def _set_cfg(self):
        # general experiment + environment setup/scene generation configs
        self.cfg = get_eval_cfg_defaults()
        config_fname = osp.join(path_util.get_ndf_config(), 'eval_cfgs', self.args.config + '.yaml')
        if osp.exists(config_fname):
            self.cfg.merge_from_file(config_fname)
        else:
            log_info('Config file %s does not exist, using defaults' % config_fname)
        self.cfg.freeze()
    
    def _set_obj_cfg(self):
        # object specific configs
        self.obj_cfg = get_obj_cfg_defaults()
        obj_config_name = osp.join(path_util.get_ndf_config(), self.args.object_class + '_obj_cfg.yaml')
        self.obj_cfg.merge_from_file(obj_config_name)
        self.obj_cfg.freeze()
    
    def _set_test_object_ids(self):
        test_shapenet_ids = np.loadtxt(osp.join(path_util.get_ndf_share(), '%s_test_object_split.txt' % self.obj_class), dtype=str).tolist()
        if self.obj_class == 'mug':
            avoid_shapenet_ids = bad_shapenet_mug_ids_list + self.cfg.MUG.AVOID_SHAPENET_IDS
        elif self.obj_class == 'bowl':
            avoid_shapenet_ids = bad_shapenet_bowls_ids_list + self.cfg.BOWL.AVOID_SHAPENET_IDS
        elif self.obj_class == 'bottle':
            avoid_shapenet_ids = bad_shapenet_bottles_ids_list + self.cfg.BOTTLE.AVOID_SHAPENET_IDS 
        else:
            test_shapenet_ids = []

        shapenet_id_list = [fn.split('_')[0] for fn in os.listdir(self.shapenet_obj_dir)] if self.obj_class == 'mug' else os.listdir(self.shapenet_obj_dir)
        for s_id in shapenet_id_list:
            valid = s_id not in self.demo_shapenet_ids and s_id not in avoid_shapenet_ids
            if args.only_test_ids:
                valid = valid and (s_id in test_shapenet_ids)
        
            if valid:
                self.test_object_ids.append(s_id)

        if self.args.single_instance:
            self.test_object_ids = [self.demo_shapenet_ids[0]]

    def _robot_config(self):
        table_z = self.cfg.TABLE_Z

        p.changeDynamics(self.robot.arm.robot_id, self.left_pad_id, lateralFriction=1.0)
        p.changeDynamics(self.robot.arm.robot_id, self.right_pad_id, lateralFriction=1.0)

        # reset robot in preparation for testing
        self.robot.arm.reset(force_reset=True)
        self.robot.cam.setup_camera(
            focus_pt=[0.4, 0.0, table_z],
            dist=0.9,
            yaw=45,
            pitch=-25,
            roll=0)

    def _set_cams(self):
        self.cams = MultiCams(self.cfg.CAMERA, self.robot.pb_client, n_cams=self.cfg.N_CAMERAS)
        self.cam_info = {}
        self.cam_info['pose_world'] = []
        for cam in self.cams.cams:
            self.cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))
    
    def _set_table(self):
        table_ori = euler2quat([0, 0, np.pi / 2])
        grasp_data = self.demo_loader.get_arbitrary_grasp_data()

        # this is the URDF that was used in the demos -- make sure we load an identical one
        tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack_tmp.urdf')
        open(tmp_urdf_fname, 'w').write(grasp_data['table_urdf'].item())
        self.table_id = self.robot.pb_client.load_urdf(tmp_urdf_fname,
                                self.cfg.TABLE_POS,
                                table_ori,
                                scaling=self.cfg.TABLE_SCALING)
        
        # print('self.table_id: ', self.table_id)
    
    def _set_rack_link_id(self):
        if self.obj_class == 'mug':
            self.rack_link_id = 0
        elif self.obj_class in {'bowl', 'bottle'}:
            self.rack_link_id = None
        else:
            raise ValueError('Unknown object type')

    def _set_shelf_link_id(self):
        if self.obj_class == 'mug':
            self.shelf_link_id = 1
        elif self.obj_class in {'bowl', 'bottle'}:
            self.shelf_link_id = 0
        else:
            raise ValueError('Unknown object type')
    
    def _set_placement_link_id(self):
        if self.cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
            self.placement_link_id = self.shelf_link_id
        else:
            self.placement_link_id = self.rack_link_id 

    @staticmethod
    def hide_link(obj_id, link_id): 
        if link_id is not None:
            p.changeVisualShape(obj_id, link_id, rgbaColor=[0, 0, 0, 0])
    
    @staticmethod
    def show_link(obj_id, link_id, color):
        if link_id is not None:
            p.changeVisualShape(obj_id, link_id, rgbaColor=color)
    
    def evaluate_single_object(self, iteration=0):
        # load a test object
        obj_shapenet_id = random.sample(self.test_object_ids, 1)[0]
        id_str = 'Shapenet ID: %s' % obj_shapenet_id
        log_info(id_str)

        viz_dict = {}  # will hold information that's useful for post-run visualizations
        eval_iter_dir = osp.join(self.eval_save_dir, 'trial_%d' % iteration)
        util.safe_makedirs(eval_iter_dir)

        # for testing, use the "normalized" object
        obj_obj_file = osp.join(shapenet_obj_dir, obj_shapenet_id, 'models/model_normalized.obj')
        obj_obj_file_dec = obj_obj_file.split('.obj')[0] + '_dec.obj'

        ### GET SCALE AND POSITION ###
        mesh_scale = self._get_mesh_scale()
        pos, ori = self._get_obj_pose()

        ### ADD STUFF TO VIZ DICT ###
        viz_dict['shapenet_id'] = obj_shapenet_id
        viz_dict['obj_obj_file'] = obj_obj_file

        if 'normalized' not in shapenet_obj_dir:
            viz_dict['obj_obj_norm_file'] = osp.join(shapenet_obj_dir + '_normalized', obj_shapenet_id, 'models/model_normalized.obj')
        else:
            viz_dict['obj_obj_norm_file'] = osp.join(shapenet_obj_dir, obj_shapenet_id, 'models/model_normalized.obj')
        viz_dict['obj_obj_file_dec'] = obj_obj_file_dec
        viz_dict['mesh_scale'] = mesh_scale

        self._convert_mesh(obj_obj_file_dec, obj_obj_file)

        ### MOVE ROBOT TO START LOCATION ###
        self._home_robot()

        ### LOAD GEOMETRY ### TODO: refactor more
        obj_id = self.robot.pb_client.load_geom(
            'mesh',
            mass=0.01,
            mesh_scale=mesh_scale,
            visualfile=obj_obj_file_dec,
            collifile=obj_obj_file_dec,
            base_pos=pos,
            base_ori=ori)
        p.changeDynamics(obj_id, -1, lateralFriction=0.5)

        if obj_class == 'bowl':
            safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=self.table_id, linkIndexA=-1, linkIndexB=self.rack_link_id, enableCollision=False)
            safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=self.table_id, linkIndexA=-1, linkIndexB=self.shelf_link_id, enableCollision=False)
            self.robot.pb_client.set_step_sim(False)

        o_cid = None
        if args.any_pose:
            o_cid = constraint_obj_world(obj_id, pos, ori)
            self.robot.pb_client.set_step_sim(False)
        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        time.sleep(1.5)

        self.hide_link(self.table_id, self.rack_link_id)


        # Get poses
        obj_pose_world = p.getBasePositionAndOrientation(obj_id)
        obj_pose_world = util.list2pose_stamped(list(obj_pose_world[0]) + list(obj_pose_world[1]))
        viz_dict['start_obj_pose'] = util.pose_stamped2list(obj_pose_world)

        # Get point cloud from cameras
        target_obj_pcd_obs = self._get_target_obj_pcd_obs(obj_id)

        # TODO: Determine what this does
        if obj_class == 'mug':
            rack_color = p.getVisualShapeData(self.table_id)[self.rack_link_id][7]
            self.show_link(self.table_id, self.rack_link_id, rack_color)

        if obj_class == 'bowl':
            for i in range(p.getNumJoints(self.robot.arm.robot_id)):
                safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=self.table_id, linkIndexA=i, linkIndexB=self.rack_link_id, enableCollision=False)
                safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=self.table_id, linkIndexA=i, linkIndexB=self.shelf_link_id, enableCollision=False)
            safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=self.table_id, linkIndexA=-1, linkIndexB=self.rack_link_id, enableCollision=False)
            safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=self.table_id, linkIndexA=-1, linkIndexB=self.shelf_link_id, enableCollision=False)
        
        # optimize grasp pose
        pre_grasp_ee_pose_mats, best_idx = self.grasp_optimizer.optimize_transform_implicit(target_obj_pcd_obs, ee=True)
        pre_grasp_ee_pose = util.pose_stamped2list(util.pose_from_matrix(pre_grasp_ee_pose_mats[best_idx]))
        viz_dict['start_ee_pose'] = pre_grasp_ee_pose


        preplace_horizontal_tf_list = self.cfg.PREPLACE_HORIZONTAL_OFFSET_TF
        preplace_horizontal_tf = util.list2pose_stamped(self.cfg.PREPLACE_HORIZONTAL_OFFSET_TF)
        preplace_offset_tf = util.list2pose_stamped(self.cfg.PREPLACE_OFFSET_TF)





        ######################
        # LAZY CODE BELOW HERE LOL



        ########################### grasp post-process #############################
        new_grasp_pt = post_process_grasp_point(pre_grasp_ee_pose, target_obj_pcd_obs, thin_feature=(not args.non_thin_feature), grasp_viz=args.grasp_viz, grasp_dist_thresh=args.grasp_dist_thresh)
        pre_grasp_ee_pose[:3] = new_grasp_pt
        pregrasp_offset_tf = get_ee_offset(ee_pose=pre_grasp_ee_pose)
        pre_pre_grasp_ee_pose = util.pose_stamped2list(
            util.transform_pose(pose_source=util.list2pose_stamped(pre_grasp_ee_pose), pose_transform=util.list2pose_stamped(pregrasp_offset_tf)))

        # optimize placement pose
        rack_pose_mats, best_rack_idx = self.place_optimizer.optimize_transform_implicit(target_obj_pcd_obs, ee=False)
        rack_relative_pose = util.pose_stamped2list(util.pose_from_matrix(rack_pose_mats[best_rack_idx]))

        ee_end_pose = util.transform_pose(pose_source=util.list2pose_stamped(pre_grasp_ee_pose), pose_transform=util.list2pose_stamped(rack_relative_pose))
        pre_ee_end_pose2 = util.transform_pose(pose_source=ee_end_pose, pose_transform=preplace_offset_tf)
        pre_ee_end_pose1 = util.transform_pose(pose_source=pre_ee_end_pose2, pose_transform=preplace_horizontal_tf)        

        ee_end_pose_list = util.pose_stamped2list(ee_end_pose)
        pre_ee_end_pose1_list = util.pose_stamped2list(pre_ee_end_pose1)
        pre_ee_end_pose2_list = util.pose_stamped2list(pre_ee_end_pose2)

        obj_start_pose = obj_pose_world
        obj_end_pose = util.transform_pose(pose_source=obj_start_pose, pose_transform=util.list2pose_stamped(rack_relative_pose))
        obj_end_pose_list = util.pose_stamped2list(obj_end_pose)
        viz_dict['final_obj_pose'] = obj_end_pose_list

        # save visualizations for debugging / looking at optimizaiton solutions
        if args.save_vis_per_model:
            analysis_dir = args.model_path + '_' + str(obj_shapenet_id)
            eval_iter_dir = osp.join(eval_save_dir, analysis_dir)
            if not osp.exists(eval_iter_dir):
                os.makedirs(eval_iter_dir)
            for f_id, fname in enumerate(self.grasp_optimizer.viz_files):
                new_viz_fname = fname.split('/')[-1]
                viz_index = int(new_viz_fname.split('.html')[0].split('_')[-1])
                new_fname = osp.join(eval_iter_dir, new_viz_fname)
                if args.save_all_opt_results:
                    shutil.copy(fname, new_fname)
                else:
                    if viz_index == best_idx:
                        print('Saving best viz_file to %s' % new_fname)
                        shutil.copy(fname, new_fname)
            for f_id, fname in enumerate(self.place_optimizer.viz_files):
                new_viz_fname = fname.split('/')[-1]
                viz_index = int(new_viz_fname.split('.html')[0].split('_')[-1])
                new_fname = osp.join(eval_iter_dir, new_viz_fname)
                if args.save_all_opt_results:
                    shutil.copy(fname, new_fname)
                else:
                    if viz_index == best_rack_idx:
                        print('Saving best viz_file to %s' % new_fname)
                        shutil.copy(fname, new_fname)
        
        self.viz_data_list.append(viz_dict)
        viz_sample_fname = osp.join(eval_iter_dir, 'overlay_visualization_data.npz')
        np.savez(viz_sample_fname, viz_dict=viz_dict, viz_data_list=self.viz_data_list)

        # reset object to placement pose to detect placement success
        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=False)
        safeCollisionFilterPair(obj_id, self.table_id, -1, self.placement_link_id, enableCollision=False)
        self.robot.pb_client.set_step_sim(True)
        safeRemoveConstraint(o_cid)
        self.robot.pb_client.reset_body(obj_id, obj_end_pose_list[:3], obj_end_pose_list[3:])

        time.sleep(1.0)
        teleport_rgb = self.robot.cam.get_images(get_rgb=True)[0]
        teleport_img_fname = osp.join(self.eval_teleport_imgs_dir, '%d.png' % iteration)
        np2img(teleport_rgb.astype(np.uint8), teleport_img_fname)
        safeCollisionFilterPair(obj_id, self.table_id, -1, self.placement_link_id, enableCollision=True)
        self.robot.pb_client.set_step_sim(False)
        time.sleep(1.0)

        obj_surf_contacts = p.getContactPoints(obj_id, self.table_id, -1, self.placement_link_id)
        touching_surf = len(obj_surf_contacts) > 0
        place_success_teleport = touching_surf
        self.place_success_teleport_list.append(place_success_teleport)

        time.sleep(1.0)
        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
        self.robot.pb_client.reset_body(obj_id, pos, ori)

        # attempt grasp and solve for plan to execute placement with arm
        jnt_pos = grasp_jnt_pos = grasp_plan = None
        place_success = grasp_success = False
        for g_idx in range(2):

            # reset everything
            self.robot.pb_client.set_step_sim(False)
            safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
            if args.any_pose:
                self.robot.pb_client.set_step_sim(True)
            safeRemoveConstraint(o_cid)
            p.resetBasePositionAndOrientation(obj_id, pos, ori)
            print(p.getBasePositionAndOrientation(obj_id))
            time.sleep(0.5)

            if args.any_pose:
                o_cid = constraint_obj_world(obj_id, pos, ori)
                self.robot.pb_client.set_step_sim(False)
            self.robot.arm.go_home(ignore_physics=True)

            # turn OFF collisions between robot and object / table, and move to pre-grasp pose
            for i in range(p.getNumJoints(self.robot.arm.robot_id)):
                safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=self.table_id, linkIndexA=i, linkIndexB=-1, enableCollision=False, physicsClientId=self.robot.pb_client.get_client_id())
                safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=False, physicsClientId=self.robot.pb_client.get_client_id())
            self.robot.arm.eetool.open()

            if jnt_pos is None or grasp_jnt_pos is None: 
                jnt_pos = self.ik_helper.get_feasible_ik(pre_pre_grasp_ee_pose)
                grasp_jnt_pos = self.ik_helper.get_feasible_ik(pre_grasp_ee_pose)

                if jnt_pos is None or grasp_jnt_pos is None:
                    jnt_pos = self.ik_helper.get_ik(pre_pre_grasp_ee_pose)
                    grasp_jnt_pos = self.ik_helper.get_ik(pre_grasp_ee_pose)

                    if jnt_pos is None or grasp_jnt_pos is None:
                        jnt_pos = self.robot.arm.compute_ik(pre_pre_grasp_ee_pose[:3], pre_pre_grasp_ee_pose[3:])
                        grasp_jnt_pos = self.robot.arm.compute_ik(pre_grasp_ee_pose[:3], pre_grasp_ee_pose[3:])  # this is the pose that's at the grasp, where we just need to close the fingers

            if grasp_jnt_pos is not None and jnt_pos is not None:
                if g_idx == 0:
                    self.robot.pb_client.set_step_sim(True)
                    self.robot.arm.set_jpos(grasp_jnt_pos, ignore_physics=True)
                    self.robot.arm.eetool.close(ignore_physics=True)
                    time.sleep(0.2)
                    grasp_rgb = self.robot.cam.get_images(get_rgb=True)[0]
                    grasp_img_fname = osp.join(self.eval_grasp_imgs_dir, '%d.png' % iteration)
                    np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)
                    continue
                
                ########################### planning to pre_pre_grasp and pre_grasp ##########################
                if grasp_plan is None:
                    plan1 = self.ik_helper.plan_joint_motion(self.robot.arm.get_jpos(), jnt_pos)
                    plan2 = self.ik_helper.plan_joint_motion(jnt_pos, grasp_jnt_pos)
                    if plan1 is not None and plan2 is not None:
                        grasp_plan = plan1 + plan2

                        self.robot.arm.eetool.open()
                        for jnt in plan1:
                            self.robot.arm.set_jpos(jnt, wait=False)
                            time.sleep(0.025)
                        self.robot.arm.set_jpos(plan1[-1], wait=True)
                        for jnt in plan2:
                            self.robot.arm.set_jpos(jnt, wait=False)
                            time.sleep(0.04)
                        self.robot.arm.set_jpos(grasp_plan[-1], wait=True)

                        # get pose that's straight up
                        offset_pose = util.transform_pose(
                            pose_source=util.list2pose_stamped(np.concatenate(self.robot.arm.get_ee_pose()[:2]).tolist()),
                            pose_transform=util.list2pose_stamped([0, 0, 0.15, 0, 0, 0, 1])
                        )
                        offset_pose_list = util.pose_stamped2list(offset_pose)
                        offset_jnts = self.ik_helper.get_feasible_ik(offset_pose_list)

                        # turn ON collisions between robot and object, and close fingers
                        for i in range(p.getNumJoints(self.robot.arm.robot_id)):
                            safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=True, physicsClientId=self.robot.pb_client.get_client_id())
                            safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=self.table_id, linkIndexA=i, linkIndexB=self.rack_link_id, enableCollision=False, physicsClientId=self.robot.pb_client.get_client_id())

                        time.sleep(0.8)
                        obj_pos_before_grasp = p.getBasePositionAndOrientation(obj_id)[0]
                        jnt_pos_before_grasp = self.robot.arm.get_jpos()
                        soft_grasp_close(self.robot, self.finger_joint_id, force=50)
                        safeRemoveConstraint(o_cid)
                        time.sleep(0.8)
                        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=False)
                        time.sleep(0.8)

                        if g_idx == 1:
                            grasp_success = object_is_still_grasped(self.robot, obj_id, self.right_pad_id, self.left_pad_id) 

                            if grasp_success:
                            # turn OFF collisions between object / table and object / rack, and move to pre-place pose
                                safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
                                self.robot.arm.eetool.open()
                                p.resetBasePositionAndOrientation(obj_id, obj_pos_before_grasp, ori)
                                soft_grasp_close(self.robot, self.finger_joint_id, force=40)
                                self.robot.arm.set_jpos(jnt_pos_before_grasp, ignore_physics=True)
                                cid = constraint_grasp_close(self.robot, obj_id)
                                
                        #########################################################################################################

                        if offset_jnts is not None:
                            offset_plan = self.ik_helper.plan_joint_motion(self.robot.arm.get_jpos(), offset_jnts)

                            if offset_plan is not None:
                                for jnt in offset_plan:
                                    self.robot.arm.set_jpos(jnt, wait=False)
                                    time.sleep(0.04)
                                self.robot.arm.set_jpos(offset_plan[-1], wait=True)

                        # turn OFF collisions between object / table and object / rack, and move to pre-place pose
                        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=False)
                        safeCollisionFilterPair(obj_id, self.table_id, -1, self.rack_link_id, enableCollision=False)
                        time.sleep(1.0)

        if grasp_success:
            ####################################### get place pose ###########################################

            pre_place_jnt_pos1 = self.ik_helper.get_feasible_ik(pre_ee_end_pose1_list)
            pre_place_jnt_pos2 = self.ik_helper.get_feasible_ik(pre_ee_end_pose2_list)
            place_jnt_pos = self.ik_helper.get_feasible_ik(ee_end_pose_list)

            if place_jnt_pos is not None and pre_place_jnt_pos2 is not None and pre_place_jnt_pos1 is not None:
                plan1 = self.ik_helper.plan_joint_motion(self.robot.arm.get_jpos(), pre_place_jnt_pos1)
                plan2 = self.ik_helper.plan_joint_motion(pre_place_jnt_pos1, pre_place_jnt_pos2)
                plan3 = self.ik_helper.plan_joint_motion(pre_place_jnt_pos2, place_jnt_pos)

                if plan1 is not None and plan2 is not None and plan3 is not None:
                    place_plan = plan1 + plan2

                    for jnt in place_plan:
                        self.robot.arm.set_jpos(jnt, wait=False)
                        time.sleep(0.035) 
                    self.robot.arm.set_jpos(place_plan[-1], wait=True)

                ################################################################################################################

                    # turn ON collisions between object and rack, and open fingers
                    safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
                    safeCollisionFilterPair(obj_id, self.table_id, -1, self.rack_link_id, enableCollision=True)

                    for jnt in plan3:
                        self.robot.arm.set_jpos(jnt, wait=False)
                        time.sleep(0.075) 
                    self.robot.arm.set_jpos(plan3[-1], wait=True)

                    p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
                    constraint_grasp_open(cid)
                    self.robot.arm.eetool.open()

                    time.sleep(0.2)
                    for i in range(p.getNumJoints(self.robot.arm.robot_id)):
                        safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=False, physicsClientId=self.robot.pb_client.get_client_id())
                    self.robot.arm.move_ee_xyz([0, 0.075, 0.075])
                    safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=False)
                    time.sleep(4.0)

                    # observe and record outcome
                    obj_surf_contacts = p.getContactPoints(obj_id, self.table_id, -1, self.placement_link_id)
                    touching_surf = len(obj_surf_contacts) > 0
                    obj_floor_contacts = p.getContactPoints(obj_id, self.robot.arm.floor_id, -1, -1)
                    touching_floor = len(obj_floor_contacts) > 0
                    place_success = touching_surf and not touching_floor

        self.robot.arm.go_home(ignore_physics=True)

        self.place_success_list.append(place_success)
        self.grasp_success_list.append(grasp_success)
        log_str = 'Iteration: %d, ' % iteration
        kvs = {}
        kvs['Place Success'] = sum(self.place_success_list) / float(len(self.place_success_list))
        kvs['Place [teleport] Success'] = sum(self.place_success_teleport_list) / float(len(self.place_success_teleport_list))
        kvs['Grasp Success'] = sum(self.grasp_success_list) / float(len(self.grasp_success_list))
        print('Place success list: ', self.place_success_list)
        for k, v in kvs.items():
            log_str += '%s: %.3f, ' % (k, v)
        id_str = 'shapenet_id: %s' % obj_shapenet_id
        log_info(log_str + id_str)

        with open(osp.join(self.eval_log_dir, self.log_fn), 'a') as f:
            f.write(log_str + id_str + '\n')

        eval_iter_dir = osp.join(eval_save_dir, 'trial_%d' % iteration)
        if not osp.exists(eval_iter_dir):
            os.makedirs(eval_iter_dir)
        sample_fname = osp.join(eval_iter_dir, 'success_rate_eval_implicit.npz')
        print('Saving eval logs to: %s' % sample_fname)
        np.savez(
            sample_fname,
            obj_shapenet_id=obj_shapenet_id,
            success=self.success_list,
            grasp_success=grasp_success,
            place_success=place_success,
            place_success_teleport=place_success_teleport,
            grasp_success_list=self.grasp_success_list,
            place_success_list=self.place_success_list,
            place_success_teleport_list=self.place_success_teleport_list,
            start_obj_pose=util.pose_stamped2list(obj_start_pose),
            best_place_obj_pose=obj_end_pose_list,
            ee_transforms=pre_grasp_ee_pose_mats,
            obj_transforms=rack_pose_mats,
            mesh_file=obj_obj_file,
            distractor_info=None,
            args=args.__dict__,
            global_dict=global_dict,
            cfg=util.cn2dict(self.cfg),
            obj_cfg=util.cn2dict(self.obj_cfg)
        )

        self.robot.pb_client.remove_body(obj_id)
    
    def _get_mesh_scale(self):
        scale_high, scale_low = self.cfg.MESH_SCALE_HIGH, self.cfg.MESH_SCALE_LOW
        scale_default = self.cfg.MESH_SCALE_DEFAULT
        if self.args.rand_mesh_scale:
            mesh_scale = [np.random.random() * (scale_high - scale_low) + scale_low] * 3
        else:
            mesh_scale=[scale_default] * 3
        
        return mesh_scale

    def _get_obj_pose(self):

        x_low, x_high = self.cfg.OBJ_SAMPLE_X_HIGH_LOW
        y_low, y_high = self.cfg.OBJ_SAMPLE_Y_HIGH_LOW
        table_z = self.cfg.TABLE_Z

        if self.obj_class in ['bottle', 'jar', 'bowl', 'mug']:
            upright_orientation = common.euler2quat([np.pi/2, 0, 0]).tolist()
        else:
            upright_orientation = common.euler2quat([0, 0, 0]).tolist()

        if self.args.any_pose:
            if obj_class in ['bowl', 'bottle']:
                rp = np.random.rand(2) * (2 * np.pi / 3) - (np.pi / 3)
                ori = common.euler2quat([rp[0], rp[1], 0]).tolist()
            else:
                rpy = np.random.rand(3) * (2 * np.pi / 3) - (np.pi / 3)
                ori = common.euler2quat([rpy[0], rpy[1], rpy[2]]).tolist()

            pos = [
                np.random.random() * (x_high - x_low) + x_low,
                np.random.random() * (y_high - y_low) + y_low,
                table_z]
            pose = pos + ori
            rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi, max_theta=np.pi)
            pose_w_yaw = util.transform_pose(util.list2pose_stamped(pose), util.pose_from_matrix(rand_yaw_T))
            pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]
        else:
            pos = [np.random.random() * (x_high - x_low) + x_low, np.random.random() * (y_high - y_low) + y_low, table_z]
            pose = util.list2pose_stamped(pos + upright_orientation)
            rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi, max_theta=np.pi)
            pose_w_yaw = util.transform_pose(pose, util.pose_from_matrix(rand_yaw_T))
            pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]
        
        return pos, ori
    
    def _convert_mesh(self, obj_obj_file_dec, obj_obj_file):
        if not osp.exists(obj_obj_file_dec):
            p.vhacd(
                obj_obj_file,
                obj_obj_file_dec,
                'log.txt',
                concavity=0.0025,
                alpha=0.04,
                beta=0.05,
                gamma=0.00125,
                minVolumePerCH=0.0001,
                resolution=1000000,
                depth=20,
                planeDownsampling=4,
                convexhullDownsampling=4,
                pca=0,
                mode=0,
                convexhullApproximation=1
            )
    
    def _home_robot(self):
        self.robot.arm.go_home(ignore_physics=True)
        self.robot.arm.move_ee_xyz([0, 0, 0.2])

        if args.any_pose:
            self.robot.pb_client.set_step_sim(True)
        if obj_class in ['bowl']:
            self.robot.pb_client.set_step_sim(True)
    
    def _get_target_obj_pcd_obs(self, obj_id):
        depth_imgs = []
        seg_idxs = []
        obj_pcd_pts = []
        table_pcd_pts = []
        rack_pcd_pts = []

        for i, cam in enumerate(self.cams.cams): 
            # get image and raw point cloud
            rgb, depth, seg = cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
            pts_raw, _ = cam.get_pcd(in_world=True, rgb_image=rgb, depth_image=depth, depth_min=0.0, depth_max=np.inf)

            # flatten and find corresponding pixels in segmentation mask
            flat_seg = seg.flatten()
            flat_depth = depth.flatten()
            obj_inds = np.where(flat_seg == obj_id)
            table_inds = np.where(flat_seg == self.table_id)
            seg_depth = flat_depth[obj_inds[0]]  

            # print('table_id: ', table_id, 'table_inds: ', table_inds, 'flat_seg: ', flat_seg)
            
            obj_pts = pts_raw[obj_inds[0], :]
            obj_pcd_pts.append(util.crop_pcd(obj_pts))
            table_pts = pts_raw[table_inds[0], :][::int(table_inds[0].shape[0]/500)]
            table_pcd_pts.append(table_pts)

            if self.rack_link_id is not None:
                rack_val = self.table_id + ((self.rack_link_id+1) << 24)
                rack_inds = np.where(flat_seg == rack_val)
                if rack_inds[0].shape[0] > 0:
                    rack_pts = pts_raw[rack_inds[0], :]
                    rack_pcd_pts.append(rack_pts)
        
            depth_imgs.append(seg_depth)
            seg_idxs.append(obj_inds)

        target_obj_pcd_obs = np.concatenate(obj_pcd_pts, axis=0)  # object shape point cloud
        target_pts_mean = np.mean(target_obj_pcd_obs, axis=0)
        inliers = np.where(np.linalg.norm(target_obj_pcd_obs - target_pts_mean, 2, 1) < 0.2)[0]
        target_obj_pcd_obs = target_obj_pcd_obs[inliers]

        return target_obj_pcd_obs

    def evaluate_ndf(self):
        for iteration in range(self.args.start_iteration, self.args.num_iterations):
            self.evaluate_single_object(iteration=iteration)
    

def main(args, global_dict):
    evaluate_ndf_helper = Evaluate_NDF(args, global_dict)

    evaluate_ndf_helper.evaluate_ndf()

    print('Finished evaluating!')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--eval_data_dir', type=str, default='eval_data')
    parser.add_argument('--demo_exp', type=str, default='debug_label')
    parser.add_argument('--exp', type=str, default='debug_eval')
    parser.add_argument('--object_class', type=str, default='mug')
    parser.add_argument('--opt_iterations', type=int, default=250)
    parser.add_argument('--num_demo', type=int, default=12, help='number of demos use')
    parser.add_argument('--any_pose', action='store_true')
    parser.add_argument('--num_iterations', type=int, default=100)
    parser.add_argument('--resume_iter', type=int, default=0)
    parser.add_argument('--config', type=str, default='base_cfg')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_vis_per_model', action='store_true')
    parser.add_argument('--noise_scale', type=float, default=0.05)
    parser.add_argument('--noise_decay', type=float, default=0.75)
    parser.add_argument('--pybullet_viz', action='store_true')
    parser.add_argument('--dgcnn', action='store_true')
    parser.add_argument('--random', action='store_true', help='utilize random weights')
    parser.add_argument('--early_weight', action='store_true', help='utilize early weights')
    parser.add_argument('--late_weight', action='store_true', help='utilize late weights')
    parser.add_argument('--rand_mesh_scale', action='store_true')
    parser.add_argument('--only_test_ids', action='store_true')
    parser.add_argument('--all_cat_model', action='store_true', help='True if we want to use a model that was trained on multipl categories')
    parser.add_argument('--n_demos', type=int, default=0, help='if some integer value greater than 0, we will only use that many demonstrations')
    parser.add_argument('--acts', type=str, default='all')
    parser.add_argument('--old_model', action='store_true', help='True if using a model using the old extents centering, else new one uses mean centering + com offset')
    parser.add_argument('--save_all_opt_results', action='store_true', help='If True, then we will save point clouds for all optimization runs, otherwise just save the best one (which we execute)')
    parser.add_argument('--grasp_viz', action='store_true') # Only works if pybullet_viz is on
    parser.add_argument('--single_instance', action='store_true')
    parser.add_argument('--non_thin_feature', action='store_true')
    parser.add_argument('--grasp_dist_thresh', type=float, default=0.0025)
    parser.add_argument('--start_iteration', type=int, default=0)

    parser.add_argument('--use_gripper_occ', action='store_true')


    args = parser.parse_args()

    signal.signal(signal.SIGINT, util.signal_handler)

    obj_class = args.object_class
    shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(), obj_class + '_centered_obj_normalized')

    demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', obj_class, args.demo_exp)

    expstr = 'exp--' + str(args.exp)
    modelstr = 'model--' + str(args.model_path)
    seedstr = 'seed--' + str(args.seed)
    occstr = 'occ--' + str(args.use_gripper_occ)

    full_experiment_name = '_'.join([expstr, modelstr, occstr, seedstr])
    eval_save_dir = osp.join(path_util.get_ndf_eval_data(), args.eval_data_dir, full_experiment_name)
    util.safe_makedirs(eval_save_dir)

    vnn_model_path = osp.join(path_util.get_ndf_model_weights(), args.model_path + '.pth')

    global_dict = dict(
        shapenet_obj_dir=shapenet_obj_dir,
        demo_load_dir=demo_load_dir,
        eval_save_dir=eval_save_dir,
        object_class=obj_class,
        vnn_checkpoint_path=vnn_model_path
    )

    # print("Demo dir", global_dict['demo_load_dir'])
    # /home/elchun/Documents/LIS/ndf_robot/src/ndf_robot/data/demos/mug/grasp_rim_hang_handle_gaussian_precise_w_shell

    main(args, global_dict)
