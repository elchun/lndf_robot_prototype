"""
New evaluate procedure to evaluate grasp ability of networks

Options:
Load different types of networks
Load different types of evaluation procedures

Structure:
Parser:
    Read config file
    Pass appropriate arguments to evaluator

Evaluator:
    Use configs to generate appropriate network
    Use configs to generate appropriate evaluator
    Copy configs to file evaluation folder
"""

from enum import Enum
import numpy as np
import os, os.path as osp
import yaml
import random
from datetime import datetime
import time

import torch
import trimesh

import pybullet as p

from airobot import Robot
from airobot.utils import common
from airobot.utils.common import euler2quat
from ndf_robot.config.default_cam_cfg import get_default_cam_cfg
from ndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from ndf_robot.config.default_obj_cfg import get_obj_cfg_defaults

from ndf_robot.utils import path_util, util

import ndf_robot.model.vnn_occupancy_net.vnn_occupancy_net_pointnet_dgcnn \
    as vnn_occupancy_network
import ndf_robot.model.conv_occupancy_net.conv_occupancy_net \
    as conv_occupancy_network

from ndf_robot.opt.optimizer import OccNetOptimizer
from ndf_robot.robot.multicam import MultiCams
from ndf_robot.utils.franka_ik import FrankaIK

from ndf_robot.utils.eval_gen_utils import (
    soft_grasp_close, constraint_grasp_close, constraint_obj_world, constraint_grasp_open,
    safeCollisionFilterPair, object_is_still_grasped, get_ee_offset, post_process_grasp_point,
    process_demo_data_rack, process_demo_data_shelf, process_xq_data, process_xq_rs_data, safeRemoveConstraint,
)


ModelTypes = {
    'CONV_OCC',
    'VNN_NDF',
}

QueryPointTypes = {
    'SPHERE'
}


class RobotIDs:
    finger_joint_id = 9
    left_pad_id = 9
    right_pad_id = 10


class SimConstants:
    # General configs
    N_CAMERAS = 4

    PREGRASP_OFFSET_TF = [0, 0, 0.25, 0, 0, 0, 1]

    # placement of table
    TABLE_POS = [0.5, 0.0, 0.4]
    TABLE_SCALING = 0.9
    TABLE_Z = 1.15

    # placement of object
    OBJ_SAMPLE_X_LOW_HIGH = [0.4, 0.5]
    OBJ_SAMPLE_Y_LOW_HIGH = [-0.4, 0.4]

    # Object scales
    MESH_SCALE_DEFAULT = 0.3
    MESH_SCALE_HIGH = 0.4
    MESH_SCALE_LOW = 0.2


class EvaluateGrasp():
    """
    Class for running evaluation on robot arm
    """

    def __init__(self, optimizer: OccNetOptimizer, seed: int,
                 shapenet_obj_dir: str, eval_save_dir: str, demo_load_dir: str,
                 pybullet_viz: bool = False, obj_class: str = 'mug'):
        self.optimizer = optimizer
        self.seed = seed

        self.robot = Robot('franka',
                           pb_cfg={'gui': pybullet_viz},
                           arm_cfg={'self_collision': False, 'seed': seed})
        self.ik_helper = FrankaIK(gui=False)
        self.obj_class = obj_class

        self.shapenet_obj_dir = shapenet_obj_dir
        self.eval_save_dir = eval_save_dir
        self.demo_load_dir = demo_load_dir
        self.eval_grasp_imgs_dir = osp.join(eval_save_dir, 'grasp_imgs')
        self.global_summary_fname = osp.join(eval_save_dir, 'global_summary.txt')

        self.test_shapenet_ids = np.loadtxt(osp.join(path_util.get_ndf_share(),
            '%s_test_object_split.txt' % obj_class), dtype=str).tolist()
        # ADD AVOID SHAPENET IDS LATER

        util.safe_makedirs(self.eval_grasp_imgs_dir)

    def configure_sim(self):
        """
        Run after demos are loaded
        """
        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.left_pad_id,
            lateralFriction=1.0)
        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.right_pad_id,
            lateralFriction=1.0)

        self.robot.arm.reset(force_reset=True)
        self.robot.cam.setup_camera(
            focus_pt=[0.4, 0.0, SimConstants.TABLE_Z],
            dist=0.9,
            yaw=45,
            pitch=-25,
            roll=0)

        # Set up cameras
        cam_cfg = get_default_cam_cfg()

        cams = MultiCams(cam_cfg, self.robot.pb_client,
                         n_cams=SimConstants.N_CAMERAS)
        cam_info = {}
        cam_info['pose_world'] = []
        for cam in cams.cams:
            cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))

        # put table at right spot
        table_ori = euler2quat([0, 0, np.pi / 2])

        # this is the URDF that was used in the demos -- make sure we load an identical one
        tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack_tmp.urdf')
        open(tmp_urdf_fname, 'w').write(self.table_urdf)
        self.table_id = self.robot.pb_client.load_urdf(tmp_urdf_fname,
            SimConstants.TABLE_POS,
            table_ori,
            scaling=SimConstants.TABLE_SCALING)

    def load_demos(self):
        """
        Load demos from self.demo_load_dir.  Add demo data to optimizer
        and save test_object_ids to self.test_object_ids
        """
        demo_fnames = os.listdir(self.demo_load_dir)
        assert len(demo_fnames), 'No demonstrations found in path: %s!' \
            % self.demo_load_dir

        grasp_demo_fnames = [osp.join(self.demo_load_dir, fn) for fn in
            demo_fnames if 'grasp_demo' in fn]

        # Can add selection of less demos here

        grasp_data_list = []
        demo_target_info_list = []
        demo_shapenet_ids = []

        # Iterate through all demos, extract relevant information and
        # prepare to pass into optimizer
        for grasp_demo_fn in grasp_demo_fnames:
            print('Loading demo from fname: %s' % grasp_demo_fn)
            grasp_data = np.load(grasp_demo_fn, allow_pickle=True)
            grasp_data_list.append(grasp_data)

            # -- Get object points -- #
            # observed shape point cloud at start
            demo_obj_pts = grasp_data['object_pointcloud']
            demo_pts_mean = np.mean(demo_obj_pts, axis=0)
            inliers = np.where(
                np.linalg.norm(demo_obj_pts - demo_pts_mean, 2, 1) < 0.2)[0]
            demo_obj_pts = demo_obj_pts[inliers]

            # -- Get query pts -- #
            demo_gripper_pts = self.optimizer.query_pts_origin
            demo_gripper_pcd = trimesh.PointCloud(demo_gripper_pts)

            # end-effector pose before grasping
            demo_ee_mat = util.matrix_from_pose(
                    util.list2pose_stamped(grasp_data['ee_pose_world']))
            demo_gripper_pcd.apply_transform(demo_ee_mat)

            # points we use to represent the gripper at their canonical pose
            # position shown in the demonstration
            demo_gripper_pts = np.asarray(demo_gripper_pcd.vertices)

            target_info = dict(
                demo_query_pts=demo_gripper_pts,
                demo_query_pts_real_shape=demo_gripper_pts,
                demo_obj_pts=demo_obj_pts,
                demo_ee_pose_world=grasp_data['ee_pose_world'],
                demo_query_pt_pose=grasp_data['gripper_contact_pose'],
                demo_obj_rel_transform=np.eye(4)
            )

            # -- Get shapenet id -- #
            shapenet_id = grasp_data['shapenet_id'].item()

            demo_target_info_list.append(target_info)
            demo_shapenet_ids.append(shapenet_id)

            # -- Get table urdf -- #
            self.table_urdf = grasp_data['table_urdf'].item()

        # -- Set demos -- #
        self.optimizer.set_demo_info(demo_target_info_list)

        # -- Get test objects -- #
        self.test_object_ids = []
        if self.obj_class == 'mug':
            shapenet_id_list = [fn.split('_')[0]
                for fn in os.listdir(self.shapenet_obj_dir)]
        else:
            shapenet_id_list = os.listdir(self.shapenet_obj_dir)

        for s_id in shapenet_id_list:
            valid = s_id not in demo_shapenet_ids and s_id  # and not in avoid shapenet ids

            if valid:
                self.test_object_ids.append(s_id)

    def run_trial(self, iteration: int = 0, rand_mesh_scale: bool = True,
                  any_pose: bool = True, thin_feature: bool = True,
                  grasp_viz: bool = True, grasp_dist_thresh: float = 0.0025):

        eval_iter_dir = osp.join(self.eval_save_dir, 'trial_%d' % iteration)
        util.safe_makedirs(eval_iter_dir)

        # -- Get and orient object -- #
        obj_shapenet_id = random.sample(self.test_object_ids, 1)[0]
        id_str = 'Shapenet ID: %s' % obj_shapenet_id

        upright_orientation = common.euler2quat([np.pi/2, 0, 0]).tolist()

        obj_fname = osp.join(self.shapenet_obj_dir, obj_shapenet_id,
            'models/model_normalized.obj')

        obj_file_dec = obj_fname.split('.obj')[0] + '_dec.obj'

        scale_low = SimConstants.MESH_SCALE_LOW
        scale_high = SimConstants.MESH_SCALE_HIGH
        scale_default = SimConstants.MESH_SCALE_DEFAULT
        if rand_mesh_scale:
            mesh_scale = [np.random.random() * (scale_low - scale_high)
                + scale_low] * 3
        else:
            mesh_scale = [scale_default] * 3

        x_low, x_high = SimConstants.OBJ_SAMPLE_X_LOW_HIGH
        y_low, y_high = SimConstants.OBJ_SAMPLE_Y_LOW_HIGH

        if any_pose:
            rp = np.random.rand(3) * (2 * np.pi / 3) - (np.pi / 3)
            ori = common.euler2quat([rp[0], rp[1], rp[2]]).tolist()

            pos = [
                np.random.random() * (x_high - x_low) + x_low,
                np.random.random() * (y_high - y_low) + y_low,
                SimConstants.TABLE_Z
            ]

            pose = pos + ori
            rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi,
                max_theta=np.pi)
            pose_w_yaw = util.transform_pose(util.list2pose_stamped(pose),
                util.pose_from_matrix(rand_yaw_T))
            pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], \
                util.pose_stamped2list(pose_w_yaw)[3:]
        else:
            pos = [np.random.random() * (x_high - x_low) + x_low,
                np.random.random() * (y_high - y_low) + y_low,
                SimConstants.TABLE_Z]
            pose = util.list2pose_stamped(pos + upright_orientation)
            rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi,
                max_theta=np.pi)
            pose_w_yaw = util.transform_pose(pose,
                util.pose_from_matrix(rand_yaw_T))
            pos = util.pose_stamped2list(pose_w_yaw)[:3]
            ori = util.pose_stamped2list(pose_w_yaw)[3:]

        # convert mesh with vhacd
        if not osp.exists(obj_file_dec):
            p.vhacd(
                obj_fname,
                obj_file_dec,
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

        # -- Run robot -- #
        self.robot.arm.go_home(ignore_physics=True)
        self.robot.arm.move_ee_xyz([0, 0, 0.2])

        if any_pose:
            self.robot.pb_client.set_step_sim(True)

        # load object
        obj_id = self.robot.pb_client.load_geom(
            'mesh',
            mass=0.01,
            mesh_scale=mesh_scale,
            visualfile=obj_file_dec,
            collifile=obj_file_dec,
            base_pos=pos,
            base_ori=ori
        )

        p.changeDynamics(obj_id, -1, lateralFriction=0.5)

        o_cid = None
        if any_pose:
            o_cid = constraint_obj_world(obj_id, pos, ori)
            self.robot.pb_client.set_step_sim(False)
        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        time.sleep(1.5)

        # -- Get object point cloud -- #
        depth_imgs = []
        seg_idxs = []
        obj_pcd_pts = []
        table_pcd_pts = []

        for i, cam in enumerate(self.cams.cams):
            # get image and raw point cloud
            rgb, depth, seg = cam.get_images(get_rgb=True, get_depth=True,
                get_seg=True)
            pts_raw, _ = cam.get_pcd(in_world=True, rgb_image=rgb,
                depth_image=depth, depth_min=0.0, depth_max=np.inf)

            # flatten and find corresponding pixels in segmentation mask
            flat_seg = seg.flatten()
            flat_depth = depth.flatten()
            obj_inds = np.where(flat_seg == obj_id)
            table_inds = np.where(flat_seg == table_id)
            seg_depth = flat_depth[obj_inds[0]]

            obj_pts = pts_raw[obj_inds[0], :]
            obj_pcd_pts.append(util.crop_pcd(obj_pts))
            table_pts = pts_raw[table_inds[0], :][::int(table_inds[0].shape[0]/500)]
            table_pcd_pts.append(table_pts)

            depth_imgs.append(seg_depth)
            seg_idxs.append(obj_inds)

        # object shape point cloud
        target_obj_pcd_obs = np.concatenate(obj_pcd_pts, axis=0)
        target_pts_mean = np.mean(target_obj_pcd_obs, axis=0)
        inliers = np.where(np.linalg.norm(
            target_obj_pcd_obs - target_pts_mean, 2, 1) < 0.2)[0]
        target_obj_pcd_obs = target_obj_pcd_obs[inliers]

        # -- Get grasp position -- #
        pre_grasp_ee_pose_mats, best_idx = self.optimizer.optimize_transform_implicit(
            target_obj_pcd_obs, ee=True)
        pre_grasp_ee_pose = util.pose_stamped2list(util.pose_from_matrix(
            pre_grasp_ee_pose_mats[best_idx]))

        # -- Post process grasp position -- #
        new_grasp_pt = post_process_grasp_point(
            pre_grasp_ee_pose,
            target_obj_pcd_obs,
            thin_feature=thin_feature,
            grasp_viz=grasp_viz,
            grasp_dist_thresh=grasp_dist_thresh)
        pre_grasp_ee_pose[:3] = new_grasp_pt
        pregrasp_offset_tf = get_ee_offset(ee_pose=pre_grasp_ee_pose)
        pre_pre_grasp_ee_pose = util.pose_stamped2list(
            util.transform_pose(
                pose_source=util.list2pose_stamped(pre_grasp_ee_pose),
                pose_transform=util.list2pose_stamped(pregrasp_offset_tf)))




    @classmethod
    def hide_link(cls, obj_id, link_id):
        if link_id is not None:
            p.changeVisualShape(obj_id, link_id, rgbaColor=[0, 0, 0, 0])

    @classmethod
    def show_link(cls, obj_id, link_id, color):
        if link_id is not None:
            p.changeVisualShape(obj_id, link_id, rgbaColor=color)


class EvaluateGraspSetup():
    """
    Set up experiment from config file
    """
    def __init__(self):
        self.config_dir = osp.join(path_util.get_ndf_eval(), 'eval_configs')

        self.evaluator_dict = None
        self.model_dict = None
        self.optimizer_dict = None
        self.query_pts_dict = None

        self.seed = None

    def load_config(self, fname: str):
        """
        Load config from yaml file with following fields:
            evaluator:
                ...

            model:
                model_type: VNN_NDF or CONV_OCC
                model_args:
                    ...

            optimizer:
                optimizer_args:
                    ...

            query_pts:
                query_pts_type: SPHERE (later add GRIPPER)
                query_pts_args:
                    ...

        Args:
            fname (str): Name of config file.  Assumes config file is in
                'eval_configs' in 'eval' folder.  Name does not include any
                path prefixes (e.g. 'default_config' is fine)

        """
        config_path = osp.join(self.config_dir, fname)
        with open(config_path, "r") as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.evaluator_dict = config_dict['evaluator']
        self.model_dict = config_dict['model']
        self.optimizer_dict = config_dict['optimizer']
        self.query_pts_dict = config_dict['query_pts']
        self.seed = config_dict['seed']

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        print(config_dict)

    def create_model(self) -> torch.nn.Module:
        """
        Create torch model from given configs

        Returns:
            torch.nn.Module: Either ConvOccNetwork or VNNOccNet
        """
        model_type = self.model_dict['type']
        model_args = self.model_dict['args']
        model_checkpoint = osp.join(path_util.get_ndf_model_weights(),
                                    self.model_dict['checkpoint'])

        assert model_type in ModelTypes, 'Invalid model type'

        if model_type == 'CONV_OCC':
            model = conv_occupancy_network.ConvolutionalOccupancyNetwork(
                **model_args)
            print('Using CONV OCC')

        elif model_type == 'VNN_NDF':
            model = vnn_occupancy_network.VNNOccNet(**model_args)
            print('USING NDF')

        model.load_state_dict(torch.load(model_checkpoint))

        print('---MODEL---\n', model)
        return model

    def create_optimizer(self, model: torch.nn.Module,
                         query_pts: np.ndarray) -> OccNetOptimizer:
        """
        Create OccNetOptimizer from given config

        Args:
            model (torch.nn.Module): Model to use in the optimizer
            query_pts (np.ndarray): Query points to use in optimizer

        Returns:
            OccNetOptimizer: Optimizer to find best grasp position
        """
        optimizer_args = self.optimizer_dict['args']
        optimizer = OccNetOptimizer(model, query_pts, **optimizer_args)
        return optimizer

    def create_query_pts(self) -> np.ndarray:
        """
        Create query points from given config

        Returns:
            np.ndarray: Query point as ndarray
        """

        query_pts_type = self.query_pts_dict['type']
        query_pts_args = self.query_pts_dict['args']

        assert query_pts_type in QueryPointTypes, 'Invalid query point type'

        if query_pts_type == 'SPHERE':
            query_pts = QueryPoints.generate_sphere(**query_pts_args)

        return query_pts

    def create_eval_dir(self, exp_desc: str='') -> str:
        """
        Create eval save dir as concatenation of current time
        and 'exp_desc'.

        Args:
            exp_desc (str, optional): Description of experiment. Defaults to ''.

        Returns:
            str: eval_save_dir.  Gives access to eval save directory
        """
        experiment_class = 'eval_grasp'
        t = datetime.now()
        time_str = t.strftime('%Y-%m-%d_%HH%MM%SS_%a')
        if exp_desc != '':
            experiment_name = time_str + '_' + exp_desc
        else:
            experiment_name = time_str + exp_desc

        eval_save_dir = osp.join(path_util.get_ndf_eval_data(),
                                 experiment_class,
                                 experiment_name)

        util.safe_makedirs(eval_save_dir)

        return eval_save_dir

    def get_demo_load_dir(self, obj_class: str='mug',
        demo_exp: str='grasp_rim_hang_handle_gaussian_precise_w_shelf') -> str:
        """
        Get directory of demos

        Args:
            obj_class (str, optional): Object class. Defaults to 'mug'.
            demo_exp (str, optional): Demo experiment name. Defaults to
                'grasp_rim_hang_handle_gaussian_precise_w_shelf'.

        Returns:
            str: Path to demo load dir
        """
        demo_load_dir = osp.join(path_util.get_ndf_data(),
                                 'demos', obj_class, demo_exp)

        return demo_load_dir

    def get_shapenet_obj_dir(self, obj_class: str='mug') -> str:
        """
        Get object dir of obj_class

        Args:
            obj_class (str, optional): Class of object (mug, bottle, bowl).
                Defaults to 'mug'.

        Returns:
            str: path to object dir
        """
        shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
                                    obj_class + '_centered_obj_normalized')

        return shapenet_obj_dir



class QueryPoints():
    @staticmethod
    def generate_sphere(n_pts: int, radius: float=0.05) -> np.ndarray:
        """
        Sample points inside sphere centered at origin with radius {radius}

        Args:
            n_pts (int): Number of point to sample.
            radius (float, optional): Radius of sphere to sample.
                Defaults to 0.05.

        Returns:
            np.ndarray: (n_pts x 3) array of query points
        """
        # http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        u = 2 * np.random.rand(n_pts, 1) - 1
        phi = 2 * np.pi * np.random.rand(n_pts, 1)
        r = radius * (np.random.rand(n_pts, 1)**(1 / 3.))
        x = r * np.cos(phi) * (1 - u**2)**0.5
        y = r * np.sin(phi) * (1 - u**2)**0.5
        z = r * u

        sphere_points = np.hstack((x, y, z))
        return sphere_points


if __name__ == '__main__':
    config_fname = 'debug_config.yml'

    setup = EvaluateGraspSetup()
    setup.load_config(config_fname)
    model = setup.create_model()
    query_pts = setup.create_query_pts()
    optimizer = setup.create_optimizer(model, query_pts)
    shapenet_obj_dir = setup.get_shapenet_obj_dir()
    eval_save_dir = setup.create_eval_dir('DEBUG')
    demo_load_dir = setup.get_demo_load_dir(obj_class='mug')

    experiment = EvaluateGrasp(optimizer=optimizer, seed=0,
        shapenet_obj_dir=shapenet_obj_dir, eval_save_dir=eval_save_dir,
        demo_load_dir=demo_load_dir, pybullet_viz=True, obj_class='mug')

    experiment.load_demos()
    experiment.configure_sim()
    experiment.run_trial(iteration=0, rand_mesh_scale=True, any_pose=True)

    print(optimizer)
