import argparse
import random
import time
from datetime import datetime
import yaml
# from typing import Callable

import os
import os.path as osp

import numpy as np
import torch
import pybullet as p

from scipy.spatial.transform import Rotation as R

from airobot import Robot
from airobot import log_info, log_warn, log_debug, set_log_level
from airobot.utils import common
from airobot.utils.common import euler2quat

import ndf_robot.model.vnn_occupancy_net.vnn_occupancy_net_pointnet_dgcnn \
    as vnn_occupancy_network
import ndf_robot.model.conv_occupancy_net.conv_occupancy_net \
    as conv_occupancy_network

from ndf_robot.opt.optimizer_lite import OccNetOptimizer
from ndf_robot.robot.multicam import MultiCams
from ndf_robot.utils.franka_ik import FrankaIK

from ndf_robot.utils import util, path_util
from ndf_robot.config.default_cam_cfg import get_default_cam_cfg
from ndf_robot.utils.eval_gen_utils import (
    soft_grasp_close, constraint_grasp_close, constraint_obj_world, constraint_grasp_open,
    safeCollisionFilterPair, object_is_still_grasped, get_ee_offset, post_process_grasp_point,
    process_demo_data_rack, process_demo_data_shelf, process_xq_data, process_xq_rs_data, safeRemoveConstraint,
)
from ndf_robot.eval.evaluate_general_types import (ExperimentTypes, ModelTypes,
    QueryPointTypes, TrialResults, RobotIDs, SimConstants, TrialData)
from ndf_robot.eval.query_points import QueryPoints
from ndf_robot.eval.demo_io import DemoIO


class EvaluateNetwork():
    """
    Parent class for running evaluations on robot arm
    """
    def __init__(self, seed: int, shapenet_obj_dir: str, eval_save_dir: str,
        demo_load_dir: str, test_obj_class: str, pybullet_viz: bool = False, num_trials: int = 200,
        include_avoid_obj: bool = True, any_pose: bool = True):

        self.robot = Robot('franka',
                           pb_cfg={'gui': pybullet_viz},
                           arm_cfg={'self_collision': False, 'seed': seed})
        self.ik_helper = FrankaIK(gui=False)

        self.shapenet_obj_dir = shapenet_obj_dir
        self.eval_save_dir = eval_save_dir
        self.demo_load_dir = demo_load_dir

        self.eval_grasp_imgs_dir = osp.join(eval_save_dir, 'grasp_imgs')
        self.global_summary_fname = osp.join(eval_save_dir, 'global_summary.txt')
        self.shapenet_id_list_fname = osp.join(eval_save_dir, 'shapenet_id_list.txt')

        util.safe_makedirs(self.eval_grasp_imgs_dir)

        self.num_trials = num_trials
        self.avoid_shapenet_ids = set()
        if not include_avoid_obj:
            self.avoid_shapenet_ids.update(SimConstants.MUG_AVOID_SHAPENET_IDS)
            self.avoid_shapenet_ids.update(SimConstants.BOWL_AVOID_SHAPENET_IDS)
            self.avoid_shapenet_ids.update(SimConstants.BOTTLE_AVOID_SHAPENET_IDS)

        self.train_shapenet_ids = set()
        self.train_shapenet_ids.update(SimConstants.MUG_TRAIN_SHAPENET_IDS)
        self.train_shapenet_ids.update(SimConstants.BOWL_TRAIN_SHAPENET_IDS)
        self.train_shapenet_ids.update(SimConstants.BOTTLE_TRAIN_SHAPENET_IDS)

        self.test_shapenet_ids_all = set()
        self.test_shapenet_ids_all.update(SimConstants.MUG_TEST_SHAPENET_IDS)
        self.test_shapenet_ids_all.update(SimConstants.BOWL_TEST_SHAPENET_IDS)
        self.test_shapenet_ids_all.update(SimConstants.BOTTLE_TEST_SHAPENET_IDS)
        self.test_shapenet_ids_all.update(SimConstants.BOWL_HANDLE_TEST_SHAPENET_IDS)
        self.test_shapenet_ids_all.update(self.avoid_shapenet_ids)

        self.any_pose = any_pose

        self.experiment_type = None

        self.obj_sample_x_low_high = SimConstants.OBJ_SAMPLE_X_LOW_HIGH
        self.obj_sample_y_low_high = SimConstants.OBJ_SAMPLE_Y_LOW_HIGH

        self.scale_low = SimConstants.MESH_SCALE_LOW
        self.scale_high = SimConstants.MESH_SCALE_HIGH
        self.scale_default = SimConstants.MESH_SCALE_DEFAULT

        self.test_obj_class = test_obj_class

    def load_demos(self):
        """
        Load demos relevant to optimizers used.

        Raises:
            NotImplementedError: Implement this!
        """
        raise NotImplementedError

    def configure_sim(self):
        """
        Configure simulation with relevant objects.

        Raises:
            NotImplementedError: Implement this!
        """
        raise NotImplementedError

    def run_trial(self):
        """
        Run a single trial for given experiment

        Raises:
            NotImplementedError: Implement this!
        """
        raise NotImplementedError

    def run_experiment(self, start_idx=0):
        """
        Run experiment of length specified in config.

        Raises:
            NotImplementedError: Implement this!
        """
        raise NotImplementedError

    def _compute_ik_cascade(self, pose: list):
        """
        Solve ik with three different ik solvers, using the next one if the
        previous one fails.

        Args:
            pose (list): [x, y, z, o_x, o_y, o_z, w].

        Returns:
            jnt_positions, TrialResults | None: Tuple where first arg
            describes the joint of panda, while second arg is None if ik was
            successful, else it is a TrialResult error code.
        """
        jnt_pos = None
        ik_found = False
        result = None
        # Try to compute ik in three different ways
        if not ik_found:
            jnt_pos = self.ik_helper.get_feasible_ik(pose)
            ik_found = jnt_pos is not None

        # Try this if the previous ik solver fails
        if not ik_found:
            result = TrialResults.GET_FEASIBLE_IK_FAILED
            jnt_pos = self.ik_helper.get_ik(pose)
            ik_found = jnt_pos is not None

        if not ik_found:
            result = TrialResults.GET_IK_FAILED
            jnt_pos = self.robot.arm.compute_ik(
                pose[:3], pose[3:])

            ik_found = jnt_pos is not None

        if not ik_found:
            print('compute_ik failed')
            result = TrialResults.COMPUTE_IK_FAILED
        else:
            result = None

        return jnt_pos, result

    def _take_image(self, fname):
        """
        Take image of robot in current  position.

        Args:
            fname (filename): Filename of image to save.
        """
        grasp_rgb = self.robot.cam.get_images(get_rgb=True)[0]
        util.np2img(grasp_rgb.astype(np.uint8), fname)

    @classmethod
    def _compute_anyrot_pose(cls, x_min: float, x_max: float, y_min: float,
        y_max: float, r: float = 0.1) -> 'tuple(list)':
        """
        Compute placement of mug for anyrot trials.  Makes most of
        the mugs physically possible to grab.  The goal is for the open
        end of the mug to be facing the inside of a sphere of radius {r}.
        The sphere is centered at a random point with x and y coordinates
        within [x_min, x_max] and [y_min, y_max], respectively. Since the center
        of the sphere is at the table height, any positions below the table are
        shifted up to table height + a small random shift.  Computed as follows:

        1. Get random orientation for mug.
        2. Transform vector [0, -r, 0] with orientation of mug to get position.
            of mug. The vector has -r in the y position because the mug starts
            with the opening facing the positive y direction.
        3. Compute random shift in x and y

        Args:
            x_min (float): min x position.
            x_max (float): max x position.
            y_min (float): min y position.
            y_max (float): max y position.
            r (float, optional): radius of sphere to place mugs on.  Defaults to
                0.1.

        Returns:
            tuple(list): (pos, ori) where pos is an xyz pose of dim (3, )
                and ori is a quaternion of dim (4, )
        """

        # Mugs init with the opening in the +y direction
        # Reference frame is same as robot
        #     If looking forward at robot, +y is to the right, +x is away,
        #     from robot, +z is up from ground.

        # To debug, use EvaluateGrasp.make_rotation_matrix

        ori_rot = R.random()

        pos_sphere = np.array([0, -r, 0])
        pos_sphere = ori_rot.apply(pos_sphere)

        # So that there is some variation in min z height
        z_center = SimConstants.TABLE_Z + random.random() * 0.05 \
            + SimConstants.OBJ_SAMPLE_Z_OFFSET

        x_offset = random.random() * (x_max - x_min) + x_min
        y_offset = random.random() * (y_max - y_min) + y_min
        pos = [
            pos_sphere[0] + x_offset,
            pos_sphere[1] + y_offset,
            max(z_center, pos_sphere[2] + z_center),
        ]

        ori = ori_rot.as_quat().tolist()

        return pos, ori

    def _insert_object(self, obj_shapenet_id: str, obj_scale: float,
        any_pose: bool) -> tuple:
        """
        Insert object described by {obj_shapenet_id} at calculated pose.
        Scales input mesh by amount defined in SimConstants.  This amount is
        constant if rand_mesh_scale is False, otherwise it varies based on
        SimConstants.  Inserts object upright if any_pose is False, otherwise
        inserts objects at random rotations and shifted according to
        compute_anyrot_pose().

        Args:
            obj_shapenet_id (str): shapenet id of object
            rand_mesh_scale (bool): True to use random scale for object
            any_pose (bool): True to pose at random orientation and somewhat
                random position.

        Returns:
            tuple: (obj simulation id, object constraint id,
                object pose, object orientation)
        """

        # So that any_pose object doesn't immediately fall
        if any_pose:
            self.robot.pb_client.set_step_sim(True)

        upright_orientation = common.euler2quat([np.pi / 2, 0, 0]).tolist()

        obj_fname = osp.join(self.shapenet_obj_dir, obj_shapenet_id,
            'models/model_normalized.obj')

        obj_file_dec = obj_fname.split('.obj')[0] + '_dec.obj'

        if obj_scale != -1:
            mesh_scale = [obj_scale * (self.scale_low - self.scale_high)
                + self.scale_low] * 3
        else:
            mesh_scale = [self.scale_default] * 3

        # x_low, x_high = SimConstants.OBJ_SAMPLE_X_LOW_HIGH
        # y_low, y_high = SimConstants.OBJ_SAMPLE_Y_LOW_HIGH

        x_low, x_high = self.obj_sample_x_low_high
        y_low, y_high = self.obj_sample_y_low_high
        r = SimConstants.OBJ_SAMPLE_R

        if any_pose:
            pos, ori = self._compute_anyrot_pose(x_low, x_high, y_low, y_high, r)

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

        # p.changeDynamics(obj_id, -1, lateralFriction=0.5)
        p.changeDynamics(obj_id, -1, lateralFriction=1.0, linearDamping=5, angularDamping=5)
        # p.changeDynamics(obj_id, -1, lateralFriction=4.0, linearDamping=5, angularDamping=5)

        o_cid = None
        if any_pose:
            o_cid = constraint_obj_world(obj_id, pos, ori)
            self.robot.pb_client.set_step_sim(False)

        return obj_id, o_cid, pos, ori

    def _get_pcd(self, obj_id: int) -> np.ndarray:
        """
        Use cameras to get point cloud.

        Args:
            obj_id (int): id of object in simulation.

        Returns:
            ndarray: Point cloud representing observed object.
        """
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
            table_inds = np.where(flat_seg == self.table_id)
            seg_depth = flat_depth[obj_inds[0]]

            obj_pts = pts_raw[obj_inds[0], :]
            obj_pcd_pts.append(util.crop_pcd(obj_pts))
            table_pts = pts_raw[table_inds[0], :][::int(table_inds[0].shape[0] / 500)]
            table_pcd_pts.append(table_pts)

            depth_imgs.append(seg_depth)
            seg_idxs.append(obj_inds)

        # object shape point cloud
        target_obj_pcd_obs = np.concatenate(obj_pcd_pts, axis=0)
        target_pts_mean = np.mean(target_obj_pcd_obs, axis=0)
        inliers = np.where(np.linalg.norm(
            target_obj_pcd_obs - target_pts_mean, 2, 1) < 0.2)[0]
        target_obj_pcd_obs = target_obj_pcd_obs[inliers]

        return target_obj_pcd_obs

    def _get_test_object_ids(self, demo_shapenet_ids: 'set[str]') -> 'list[str]':
        """
        Find all object that we can test on.

        Args:
            demo_shapenet_ids (set[str]): Set of ids of objects used in demos
                (to be excluded from test)

        Returns:
            list[str]: List of objects to test on.
        """
        test_object_ids = []
        shapenet_id_list = [fn.split('_')[0]
            for fn in os.listdir(self.shapenet_obj_dir)]

        for s_id in shapenet_id_list:
            valid = s_id not in demo_shapenet_ids \
                and s_id not in self.avoid_shapenet_ids \
                and s_id not in self.train_shapenet_ids \
                and s_id in self.test_shapenet_ids_all

            if valid:
                test_object_ids.append(s_id)

        return test_object_ids

    def _set_up_cameras(self):
        """
        Quick helper to put cameras in standard place.
        """
        self.robot.cam.setup_camera(
            focus_pt=[0.4, 0.0, SimConstants.CAMERA_FOCAL_Z],
            dist=0.9,
            yaw=45,
            pitch=-25,
            roll=0)

        cam_cfg = get_default_cam_cfg()

        self.cams = MultiCams(cam_cfg, self.robot.pb_client,
                         n_cams=SimConstants.N_CAMERAS)
        cam_info = {}
        cam_info['pose_world'] = []
        for cam in self.cams.cams:
            cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))
        return cam_info

    def _get_xyz_transform(self, x: float, y: float, z: float):
        return [x, y, z, 0, 0, 0, 1]


class EvaluateGrasp(EvaluateNetwork):
    def __init__(self, grasp_optimizer: OccNetOptimizer,
                 seed: int, shapenet_obj_dir: str, eval_save_dir: str,
                 demo_load_dir: str, pybullet_viz: bool = False,
                 test_obj_class: str = 'mug', num_trials: int = 200,
                 include_avoid_obj: bool = True, any_pose: bool = True):

        super().__init__(seed, shapenet_obj_dir, eval_save_dir,
            demo_load_dir, test_obj_class, pybullet_viz, num_trials,
            include_avoid_obj, any_pose)

        print(f'avoid obj: {include_avoid_obj}')
        self.grasp_optimizer = grasp_optimizer
        self.experiment_type = ExperimentTypes.GRASP

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
        demo_shapenet_ids = []

        # Iterate through all demos, extract relevant information and
        # prepare to pass into optimizer
        for grasp_demo_fn in grasp_demo_fnames:
            print('Loading grasp demo from fname: %s' % grasp_demo_fn)
            grasp_data = np.load(grasp_demo_fn, allow_pickle=True)

            demo = DemoIO.process_grasp_data(grasp_data)

            self.grasp_optimizer.add_demo(demo)
            demo_shapenet_ids.append(demo.obj_shapenet_id)

        self.grasp_optimizer.process_demos()

        # -- Get table urdf -- #
        grasp_data = np.load(grasp_demo_fnames[0], allow_pickle=True)
        self.table_urdf = DemoIO.get_table_urdf(grasp_data)

        # -- Get test objects -- #
        self.test_object_ids = self._get_test_object_ids(demo_shapenet_ids)

    def configure_sim(self):
        """
        Run after demos are loaded
        """
        # set_log_level('info')
        set_log_level('debug')

        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.left_pad_id,
            lateralFriction=1.0)
        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.right_pad_id,
            lateralFriction=1.0)

        self.robot.arm.reset(force_reset=True)
        self.robot.cam.setup_camera(
            focus_pt=[0.4, 0.0, SimConstants.CAMERA_FOCAL_Z],
            dist=0.9,
            yaw=45,
            pitch=-25,
            roll=0)

        # Set up cameras
        self._set_up_cameras()
        # cam_cfg = get_default_cam_cfg()

        # self.cams = MultiCams(cam_cfg, self.robot.pb_client,
        #                  n_cams=SimConstants.N_CAMERAS)
        # cam_info = {}
        # cam_info['pose_world'] = []
        # for cam in self.cams.cams:
        #     cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))

        # put table at right spot
        table_ori = euler2quat([0, 0, np.pi / 2])

        # Get raw table urdf
        table_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table.urdf')
        # table_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack.urdf')
        with open(table_urdf_fname, 'r', encoding='utf-8') as f:
            self.table_urdf = f.read()

        # this is the URDF that was used in the demos -- make sure we load an identical one
        tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack_tmp.urdf')
        open(tmp_urdf_fname, 'w').write(self.table_urdf)
        self.table_id = self.robot.pb_client.load_urdf(tmp_urdf_fname,
            SimConstants.TABLE_POS,
            table_ori,
            scaling=SimConstants.TABLE_SCALING)

    def run_trial(self, iteration: int = 0, obj_scale: float = -1,
                  any_pose: bool = True, thin_feature: bool = True,
                  grasp_viz: bool = False,
                  grasp_dist_thresh: float = 0.0025,
                  obj_shapenet_id: 'str | None' = None) -> TrialData:
        """
        Run trial where we try to grab object.

        Args:
            iteration (int, optional): What iteration the trial is. Defaults to 0.
            rand_mesh_scale (bool, optional): True to randomly scale mesh.
                Defaults to True.
            any_pose (bool, optional): True to use anypose function to pose mug.
                Defaults to True.
            thin_feature (bool, optional): True to treat object as thin feature
                in grasp post process. Defaults to True.
            grasp_viz (bool, optional): True to show image of grasp before trial
                runs. Only works when pybullet_viz is enabled. Defaults to False.
            grasp_dist_thresh (float, optional): Threshold to detect successful
                grasp. Defaults to 0.0025.
            obj_shapenet_id (str | None, optional): Object id to use.  If none,
                will randomly select id.

        Returns:
            TrialData: Class for storing relevant info about the trial
        """
        trial_data = TrialData()
        trial_data.aux_data = {
            'grasp_opt_idx': None,
        }

        # -- Get and orient object -- #
        if obj_shapenet_id is None:
            obj_shapenet_id = random.sample(self.test_object_ids, 1)[0]
            log_info('Generate random obj id.')
        else:
            log_info('Using predefined obj id.')
        trial_data.obj_shapenet_id = obj_shapenet_id

        # Write at start so id is recorded regardless of any later bugs
        with open(self.shapenet_id_list_fname, 'a') as f:
            f.write(f'{trial_data.obj_shapenet_id}\n')

        # -- Home Robot -- #
        self.robot.arm.go_home(ignore_physics=True)
        self.robot.arm.move_ee_xyz([0, 0, 0.2])

        # -- load object -- #
        obj_id, o_cid, pos, ori = self._insert_object(obj_shapenet_id,
            obj_scale, any_pose)

        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        time.sleep(1.5)

        # -- Get object point cloud from cameras -- #
        target_obj_pcd_obs = self._get_pcd(obj_id)

        eval_iter_dir = osp.join(self.eval_save_dir, 'trial_%s' % str(iteration).zfill(3))
        util.safe_makedirs(eval_iter_dir)

        if target_obj_pcd_obs is None or target_obj_pcd_obs.shape[0] == 0:
            trial_data.trial_result = TrialResults.GET_PCD_FAILED
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        # -- Get grasp position -- #
        log_debug('Getting grasp position.')
        opt_viz_path = osp.join(eval_iter_dir, 'visualize')
        grasp_ee_pose_mats, best_grasp_idx = self.grasp_optimizer.optimize_transform_implicit(
            target_obj_pcd_obs, ee=True, viz_path=opt_viz_path)
        grasp_ee_pose = util.pose_stamped2list(util.pose_from_matrix(
            grasp_ee_pose_mats[best_grasp_idx]))
        trial_data.aux_data['grasp_opt_idx'] = best_grasp_idx

        # -- Post process grasp position -- #
        try:
            # When there are no nearby grasp points, this throws an index
            # error.  The try catch allows us to run more trials after the error.
            new_grasp_pt = post_process_grasp_point(
                grasp_ee_pose,
                target_obj_pcd_obs,
                thin_feature=thin_feature,
                grasp_viz=grasp_viz,
                grasp_dist_thresh=grasp_dist_thresh)
        except IndexError:
            trial_data.trial_result = TrialResults.POST_PROCESS_FAILED
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        grasp_ee_pose[:3] = new_grasp_pt

        # -- Create pose which offsets gripper from object -- #
        pregrasp_offset_tf = get_ee_offset(ee_pose=grasp_ee_pose)
        pre_grasp_ee_pose = util.pose_stamped2list(
            util.transform_pose(
                pose_source=util.list2pose_stamped(grasp_ee_pose),
                pose_transform=util.list2pose_stamped(pregrasp_offset_tf)))

        # -- Get ik -- #
        log_debug('Getting ik.')
        pre_grasp_jnt_pos = grasp_jnt_pos = None

        pre_grasp_jnt_pos, ik_res = self._compute_ik_cascade(pre_grasp_ee_pose)
        ik_success = ik_res is None
        if ik_success:
            grasp_jnt_pos, ik_res = self._compute_ik_cascade(grasp_ee_pose)
            ik_success = ik_success and (ik_res is None)

        if not ik_success:
            trial_data.trial_result = ik_res
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        # -- Attempt grasp -- #
        log_debug('Attempting grasp.')
        grasp_success = False

        # turn OFF collisions between robot and object / table, and move to pre-grasp pose
        for i in range(p.getNumJoints(self.robot.arm.robot_id)):
            safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id,
                bodyUniqueIdB=self.table_id,
                linkIndexA=i,
                linkIndexB=-1,
                enableCollision=False,
                physicsClientId=self.robot.pb_client.get_client_id())

            safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id,
                bodyUniqueIdB=obj_id,
                linkIndexA=i,
                linkIndexB=-1,
                enableCollision=False,
                physicsClientId=self.robot.pb_client.get_client_id())

        home_jnt_pos = self.robot.arm.get_jpos()
        self.robot.arm.eetool.open()

        # -- Get grasp image -- #
        self.robot.pb_client.set_step_sim(True)
        self.robot.arm.set_jpos(grasp_jnt_pos, ignore_physics=True)
        self.robot.arm.eetool.close(ignore_physics=True)
        time.sleep(0.2)
        # grasp_rgb = self.robot.cam.get_images(get_rgb=True)[0]
        grasp_img_fname = osp.join(self.eval_grasp_imgs_dir,
            '%s_pose.png' % str(iteration).zfill(3))
        # util.np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)
        self._take_image(grasp_img_fname)
        self.robot.arm.eetool.open(ignore_physics=True)
        self.robot.arm.go_home(ignore_physics=True)
        time.sleep(0.5)
        self.robot.pb_client.set_step_sim(False)

        # Get to pre grasp location
        plan1 = self.ik_helper.plan_joint_motion(home_jnt_pos, pre_grasp_jnt_pos)

        # Get to grasp location
        plan2 = self.ik_helper.plan_joint_motion(pre_grasp_jnt_pos, grasp_jnt_pos)

        # Return to home location (for checking if grasp was valid)
        plan3 = self.ik_helper.plan_joint_motion(grasp_jnt_pos, home_jnt_pos)

        if None in [plan1, plan2, plan3]:
            trial_data.trial_result = TrialResults.JOINT_PLAN_FAILED
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        self.robot.arm.eetool.open()
        # Go to clearance location (linearly away from grasp area)
        for jnt in plan1:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.025)
        self.robot.arm.set_jpos(plan1[-1], wait=False)

        # Used to be below plan2 part

        # turn ON collisions between robot and object
        for i in range(p.getNumJoints(self.robot.arm.robot_id)):
            safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id,
                bodyUniqueIdB=obj_id,
                linkIndexA=i,
                linkIndexB=-1,
                enableCollision=True,
                physicsClientId=self.robot.pb_client.get_client_id())

        # Go to grasp location
        for jnt in plan2:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.04)
        self.robot.arm.set_jpos(plan2[-1], wait=False)

        time.sleep(0.8)

        # grasp_rgb = self.robot.cam.get_images(get_rgb=True)[0]
        grasp_img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_grasp.png')
        self._take_image(grasp_img_fname)
        # util.np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)

        soft_grasp_close(self.robot, RobotIDs.finger_joint_id, force=50)
        safeRemoveConstraint(o_cid)
        time.sleep(0.8)
        safeCollisionFilterPair(obj_id, self.table_id, -1, -1,
            enableCollision=False)
        time.sleep(0.8)

        for jnt in plan3:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.025)
        self.robot.arm.set_jpos(plan3[-1], wait=False)
        time.sleep(1)

        # -- Determine if grasp was successful -- #
        original_grasp_success = object_is_still_grasped(self.robot,
            obj_id, RobotIDs.right_pad_id, RobotIDs.left_pad_id)

        time.sleep(0.5)

        # If the ee was intersecting the mug, original_grasp_success
        # would be true after the table disappears.  However, an
        # intersection is generally a false grasp When the ee is
        # opened again, a good grasp should fall down while an
        # intersecting grasp would stay in contact.

        # -- Take image of grasp at clearance height -- #
        # grasp_rgb = self.robot.cam.get_images(get_rgb=True)[0]
        # util.np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)
        grasp_img_fname = osp.join(self.eval_grasp_imgs_dir,
            '%s_clearance.png' % str(iteration).zfill(3))
        self._take_image(grasp_img_fname)

        self.robot.arm.eetool.open()
        time.sleep(1)
        ee_intersecting_mug = object_is_still_grasped(
            self.robot, obj_id, RobotIDs.right_pad_id,
            RobotIDs.left_pad_id)

        grasp_success = original_grasp_success and not ee_intersecting_mug

        if ee_intersecting_mug:
            print('Intersecting grasp detected')
            trial_data.trial_result = TrialResults.INTERSECTING_EE
        else:
            if not grasp_success:
                trial_data.trial_result = TrialResults.BAD_OPT_POS

        log_info(f'Grasp success: {grasp_success}')

        if grasp_success:
            trial_data.trial_result = TrialResults.SUCCESS

        self.robot.pb_client.remove_body(obj_id)
        return trial_data

    def run_experiment(self, rand_mesh_scale: bool = True, start_idx: bool = 0):
        """
        Run experiment for {self.num_trials}
        """
        num_success = 0

        obj_shapenet_id_list = random.choices(self.test_object_ids, k=self.num_trials)

        if self.test_obj_class == 'bottle':
            thin_feature = False
        else:
            thin_feature = True

        if rand_mesh_scale:
            obj_scale_list = np.random.random(self.num_trials).tolist()
        else:
            obj_scale_list = -1 * np.ones(self.num_trials)
            obj_scale_list = obj_scale_list.tolist()

        for it in range(start_idx, self.num_trials):
            obj_shapenet_id = obj_shapenet_id_list[it]
            obj_scale = obj_scale_list[it]
            trial_data: TrialData = self.run_trial(iteration=it,
                obj_scale=obj_scale, any_pose=self.any_pose,
                obj_shapenet_id=obj_shapenet_id, thin_feature=thin_feature)

            trial_result = trial_data.trial_result
            obj_shapenet_id = trial_data.obj_shapenet_id
            best_opt_idx = trial_data.aux_data['grasp_opt_idx']

            if trial_result == TrialResults.SUCCESS:
                num_success += 1

            log_info(f'Experiment: {self.experiment_type}')
            log_info(f'Trial result: {trial_result}')
            log_info(f'Shapenet id: {obj_shapenet_id}')
            log_str = f'Successes: {num_success} | Trials {it + 1} | ' \
                + f'Success Rate: {num_success / (it + 1):0.3f}'
            log_info(log_str)

            with open(self.global_summary_fname, 'a') as f:
                f.write(f'Trial number: {it}\n')
                f.write(f'Trial result: {trial_result}\n')
                f.write(f'Grasp Success Rate: {num_success / (it + 1): 0.3f}\n')
                f.write(f'Shapenet id: {obj_shapenet_id}\n')
                f.write(f'Best Grasp idx: {best_opt_idx}\n')
                f.write('\n')


class EvaluateRackPlaceTeleport(EvaluateNetwork):
    def __init__(self, place_optimizer: OccNetOptimizer,
                 seed: int, shapenet_obj_dir: str, eval_save_dir: str,
                 demo_load_dir: str, pybullet_viz: bool = False,
                 test_obj_class: str = 'mug', num_trials: int = 200,
                 include_avoid_obj: bool = True, any_pose: bool = True):

        super().__init__(seed, shapenet_obj_dir, eval_save_dir,
            demo_load_dir, test_obj_class, pybullet_viz, num_trials,
            include_avoid_obj, any_pose)

        self.place_optimizer = place_optimizer
        self.experiment_type = ExperimentTypes.RACK_PLACE_TELEPORT

    def load_demos(self):
        """
        Load demos from self.demo_load_dir.  Add demo data to optimizer
        and save test_object_ids to self.test_object_ids
        """
        demo_fnames = os.listdir(self.demo_load_dir)
        assert len(demo_fnames), 'No demonstrations found in path: %s!' \
            % self.demo_load_dir

        place_demo_fnames = [osp.join(self.demo_load_dir, fn) for fn in
            demo_fnames if 'place_demo' in fn]

        # Can add selection of less demos here
        demo_shapenet_ids = []

        for place_demo_fn in place_demo_fnames:
            print('Loading place demo from fname: %s' % place_demo_fn)
            place_data = np.load(place_demo_fn, allow_pickle=True)

            demo = DemoIO.process_rack_place_data(place_data)

            self.place_optimizer.add_demo(demo)
            demo_shapenet_ids.append(demo.obj_shapenet_id)

        self.place_optimizer.process_demos()

        # -- Get urdf -- #
        place_data = np.load(place_demo_fnames[0], allow_pickle=True)
        self.table_urdf = DemoIO.get_table_urdf(place_data)
        self.rack_pose = DemoIO.get_rack_pose(place_data)

        # -- Get test objects -- #
        self.test_object_ids = self._get_test_object_ids(demo_shapenet_ids)

    def configure_sim(self):
        """
        Run after demos are loaded
        """
        set_log_level('debug')

        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.left_pad_id,
            lateralFriction=1.0)
        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.right_pad_id,
            lateralFriction=1.0)

        self.robot.arm.reset(force_reset=True)

        cam_info = self._set_up_cameras()

        # put table at right spot
        table_ori = euler2quat([0, 0, np.pi / 2])

        # Get raw table urdf
        # table_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table.urdf')
        table_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack.urdf')
        with open(table_urdf_fname, 'r', encoding='utf-8') as f:
            self.table_urdf = f.read()

        # this is the URDF that was used in the demos -- make sure we load an identical one
        tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack_tmp.urdf')
        open(tmp_urdf_fname, 'w').write(self.table_urdf)
        self.table_id = self.robot.pb_client.load_urdf(tmp_urdf_fname,
            SimConstants.TABLE_POS,
            table_ori,
            scaling=SimConstants.TABLE_SCALING)

    def run_trial(self, iteration: int = 0, obj_scale: float = -1,
                  any_pose: bool = True, thin_feature: bool = True,
                  grasp_viz: bool = False,
                  grasp_dist_thresh: float = 0.0025,
                  obj_shapenet_id: 'str | None' = None) -> TrialData:
        trial_data = TrialData()
        trial_data.aux_data = {
            'place_opt_idx': None,
        }

        # -- Get and orient object -- #
        if obj_shapenet_id is None:
            obj_shapenet_id = random.sample(self.test_object_ids, 1)[0]
            log_info('Generate random obj id.')
        else:
            log_info('Using predefined obj id.')
        trial_data.obj_shapenet_id = obj_shapenet_id

        # Write at start so id is recorded regardless of any later bugs
        with open(self.shapenet_id_list_fname, 'a') as f:
            f.write(f'{trial_data.obj_shapenet_id}\n')

        # -- Home Robot -- #
        self.robot.arm.go_home(ignore_physics=True)
        self.robot.arm.move_ee_xyz([0, 0, 0.2])

        # -- load object -- #
        obj_id, o_cid, pos, ori = self._insert_object(obj_shapenet_id,
            obj_scale, any_pose)

        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        time.sleep(1.5)

        # -- Get object point cloud from cameras -- #
        target_obj_pcd_obs = self._get_pcd(obj_id)

        eval_iter_dir = osp.join(self.eval_save_dir, 'trial_%s' % str(iteration).zfill(3))
        util.safe_makedirs(eval_iter_dir)

        if target_obj_pcd_obs is None or target_obj_pcd_obs.shape[0] == 0:
            trial_data.trial_result = TrialResults.GET_PCD_FAILED
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        # -- Get place position -- #
        opt_viz_path = osp.join(eval_iter_dir, 'visualize')
        rack_pose_mats, best_place_idx = self.place_optimizer.optimize_transform_implicit(
            target_obj_pcd_obs, ee=False, viz_path=opt_viz_path)
        trial_data.aux_data['place_opt_idx'] = best_place_idx
        rack_relative_pose = util.transform_pose(
            util.pose_from_matrix(rack_pose_mats[best_place_idx]), util.list2pose_stamped(self.rack_pose))

        # -- Try place teleport -- #
        obj_pose_world = p.getBasePositionAndOrientation(obj_id)
        obj_pose_world = util.list2pose_stamped(list(obj_pose_world[0]) + list(obj_pose_world[1]))
        obj_end_pose = util.transform_pose(obj_pose_world, rack_relative_pose)
        obj_end_pose = util.pose_stamped2list(obj_end_pose)
        placement_link_id = 0

        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=False)
        safeCollisionFilterPair(obj_id, self.table_id, -1, placement_link_id, enableCollision=False)
        self.robot.pb_client.set_step_sim(True)
        safeRemoveConstraint(o_cid)
        self.robot.pb_client.reset_body(obj_id, obj_end_pose[:3], obj_end_pose[3:])

        # First image suspends object in air, second is when constraints are
        # removed.
        time.sleep(1.0)
        teleport_img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_teleport_place_1.png')
        self._take_image(teleport_img_fname)
        safeCollisionFilterPair(obj_id, self.table_id, -1, placement_link_id, enableCollision=True)
        self.robot.pb_client.set_step_sim(False)
        time.sleep(1.0)
        teleport_img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_teleport_place_2.png')
        self._take_image(teleport_img_fname)

        # -- Check teleport was successful -- #
        obj_surf_contacts = p.getContactPoints(obj_id, self.table_id, -1, placement_link_id)
        touching_surf = len(obj_surf_contacts) > 0
        place_success_teleport = touching_surf
        if place_success_teleport:
            trial_data.trial_result = TrialResults.SUCCESS
        else:
            trial_data.trial_result = TrialResults.BAD_OPT_POS

        self.robot.pb_client.remove_body(obj_id)
        return trial_data

    def run_experiment(self, rand_mesh_scale: bool = True):
        """
        Run experiment for {self.num_trials}
        """
        num_success = 0

        obj_shapenet_id_list = random.choices(self.test_object_ids, k=self.num_trials)

        if rand_mesh_scale:
            obj_scale_list = np.random.random(self.num_trials).tolist()
        else:
            obj_scale_list = -1 * np.ones(self.num_trials)
            obj_scale_list = obj_scale_list.tolist()

        for it in range(self.num_trials):
            obj_shapenet_id = obj_shapenet_id_list[it]
            obj_scale = obj_scale_list[it]
            trial_data: TrialData = self.run_trial(iteration=it,
                obj_scale=obj_scale, any_pose=self.any_pose,
                obj_shapenet_id=obj_shapenet_id)

            trial_result = trial_data.trial_result
            obj_shapenet_id = trial_data.obj_shapenet_id
            best_opt_idx = trial_data.aux_data['place_opt_idx']

            if trial_result == TrialResults.SUCCESS:
                num_success += 1

            log_info(f'Experiment: {self.experiment_type}')
            log_info(f'Trial result: {trial_result}')
            log_info(f'Shapenet id: {obj_shapenet_id}')
            log_str = f'Successes: {num_success} | Trials {it + 1} | ' \
                + f'Success Rate: {num_success / (it + 1):0.3f}'
            log_info(log_str)

            with open(self.global_summary_fname, 'a') as f:
                f.write(f'Trial number: {it}\n')
                f.write(f'Trial result: {trial_result}\n')
                f.write(f'Place teleport Success Rate: {num_success / (it + 1): 0.3f}\n')
                f.write(f'Shapenet id: {obj_shapenet_id}\n')
                f.write(f'Best Grasp idx: {best_opt_idx}\n')
                f.write('\n')


class EvaluateShelfPlaceTeleport(EvaluateNetwork):
    def __init__(self, place_optimizer: OccNetOptimizer,
                 seed: int, shapenet_obj_dir: str, eval_save_dir: str,
                 demo_load_dir: str, pybullet_viz: bool = False,
                 test_obj_class: str = 'mug', num_trials: int = 200,
                 include_avoid_obj: bool = True, any_pose: bool = True):

        super().__init__(seed, shapenet_obj_dir, eval_save_dir,
            demo_load_dir, test_obj_class, pybullet_viz, num_trials,
            include_avoid_obj, any_pose)

        self.place_optimizer = place_optimizer
        self.experiment_type = ExperimentTypes.SHELF_PLACE_TELEPORT

        self.scale_low = SimConstants.MESH_SCALE_LOW
        self.scale_high = 0.5

    def load_demos(self):
        """
        Load demos from self.demo_load_dir.  Add demo data to optimizer
        and save test_object_ids to self.test_object_ids
        """
        demo_fnames = os.listdir(self.demo_load_dir)
        assert len(demo_fnames), 'No demonstrations found in path: %s!' \
            % self.demo_load_dir

        place_demo_fnames = [osp.join(self.demo_load_dir, fn) for fn in
            demo_fnames if 'place_demo' in fn]

        # Can add selection of less demos here
        demo_shapenet_ids = []

        for place_demo_fn in place_demo_fnames:
            print('Loading place demo from fname: %s' % place_demo_fn)
            place_data = np.load(place_demo_fn, allow_pickle=True)

            demo = DemoIO.process_shelf_place_data(place_data)

            self.place_optimizer.add_demo(demo)
            demo_shapenet_ids.append(demo.obj_shapenet_id)

        self.place_optimizer.process_demos()

        # -- Get urdf -- #
        place_data = np.load(place_demo_fnames[0], allow_pickle=True)
        self.table_urdf = DemoIO.get_table_urdf(place_data)
        self.shelf_pose = DemoIO.get_shelf_pose(place_data)

        # -- Get test objects -- #
        self.test_object_ids = self._get_test_object_ids(demo_shapenet_ids)

    def configure_sim(self):
        """
        Run after demos are loaded
        """
        set_log_level('debug')

        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.left_pad_id,
            lateralFriction=1.0)
        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.right_pad_id,
            lateralFriction=1.0)

        self.robot.arm.reset(force_reset=True)
        self._set_up_cameras()

        # put table at right spot
        table_ori = euler2quat([0, 0, np.pi / 2])

        # table_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_shelf.urdf')
        # with open(table_urdf_fname, 'r', encoding='utf-8') as f:
        #     self.table_urdf = f.read()

        # Write urdf from demo to temp file.
        tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack_tmp.urdf')
        open(tmp_urdf_fname, 'w').write(self.table_urdf)
        self.table_id = self.robot.pb_client.load_urdf(tmp_urdf_fname,
            SimConstants.TABLE_POS,
            table_ori,
            scaling=SimConstants.TABLE_SCALING)

    def run_trial(self, iteration: int = 0, obj_scale: float = -1,
                  any_pose: bool = True, thin_feature: bool = True,
                  grasp_viz: bool = False,
                  grasp_dist_thresh: float = 0.0025,
                  obj_shapenet_id: 'str | None' = None) -> TrialData:
        trial_data = TrialData()
        trial_data.aux_data = {
            'place_opt_idx': None,
        }

        # -- Get and orient object -- #
        if obj_shapenet_id is None:
            obj_shapenet_id = random.sample(self.test_object_ids, 1)[0]
            log_info('Generate random obj id.')
        else:
            log_info('Using predefined obj id.')
        trial_data.obj_shapenet_id = obj_shapenet_id

        # Write at start so id is recorded regardless of any later bugs
        with open(self.shapenet_id_list_fname, 'a') as f:
            f.write(f'{trial_data.obj_shapenet_id}\n')

        # -- Home Robot -- #
        self.robot.arm.go_home(ignore_physics=True)
        self.robot.arm.move_ee_xyz([0, 0, 0.2])

        # -- load object -- #
        obj_id, o_cid, pos, ori = self._insert_object(obj_shapenet_id,
            obj_scale, any_pose)

        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        time.sleep(1.5)

        img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_00ori.png')
        self._take_image(img_fname)

        # -- Get object point cloud from cameras -- #
        target_obj_pcd_obs = self._get_pcd(obj_id)

        eval_iter_dir = osp.join(self.eval_save_dir, 'trial_%s' % str(iteration).zfill(3))
        util.safe_makedirs(eval_iter_dir)

        if target_obj_pcd_obs is None or target_obj_pcd_obs.shape[0] == 0:
            trial_data.trial_result = TrialResults.GET_PCD_FAILED
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        # -- Get place position -- #
        opt_viz_path = osp.join(eval_iter_dir, 'visualize')
        pose_mats, best_opt_idx = self.place_optimizer.optimize_transform_implicit(
            target_obj_pcd_obs, ee=False, viz_path=opt_viz_path)
        trial_data.aux_data['place_opt_idx'] = best_opt_idx
        relative_pose = util.transform_pose(
            util.pose_from_matrix(pose_mats[best_opt_idx]), util.list2pose_stamped(self.shelf_pose))

        # -- Try place teleport -- #
        obj_pose_world = p.getBasePositionAndOrientation(obj_id)
        obj_pose_world = util.list2pose_stamped(list(obj_pose_world[0]) + list(obj_pose_world[1]))
        obj_end_pose = util.transform_pose(obj_pose_world, relative_pose)
        obj_end_pose = util.pose_stamped2list(obj_end_pose)
        placement_link_id = 0

        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=False)
        safeCollisionFilterPair(obj_id, self.table_id, -1, placement_link_id, enableCollision=False)
        self.robot.pb_client.set_step_sim(True)
        safeRemoveConstraint(o_cid)
        self.robot.pb_client.reset_body(obj_id, obj_end_pose[:3], obj_end_pose[3:])

        # First image suspends object in air, second is when constraints are
        # removed.
        time.sleep(1.0)
        teleport_img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_teleport_place_1.png')
        self._take_image(teleport_img_fname)
        safeCollisionFilterPair(obj_id, self.table_id, -1, placement_link_id, enableCollision=True)
        self.robot.pb_client.set_step_sim(False)
        time.sleep(1.0)
        teleport_img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_teleport_place_2.png')
        self._take_image(teleport_img_fname)

        # -- Check teleport was successful -- #
        obj_surf_contacts = p.getContactPoints(obj_id, self.table_id, -1, placement_link_id)
        touching_surf = len(obj_surf_contacts) > 0
        place_success_teleport = touching_surf
        if place_success_teleport:
            trial_data.trial_result = TrialResults.SUCCESS
        else:
            trial_data.trial_result = TrialResults.BAD_OPT_POS

        self.robot.pb_client.remove_body(obj_id)
        return trial_data

    def run_experiment(self, rand_mesh_scale: bool = True, start_idx: int = 0):
        """
        Run experiment for {self.num_trials}
        """
        num_success = 0

        obj_shapenet_id_list = random.choices(self.test_object_ids, k=self.num_trials)

        if rand_mesh_scale:
            obj_scale_list = np.random.random(self.num_trials).tolist()
        else:
            obj_scale_list = -1 * np.ones(self.num_trials)
            obj_scale_list = obj_scale_list.tolist()

        for it in range(start_idx, self.num_trials):
            obj_shapenet_id = obj_shapenet_id_list[it]
            obj_scale = obj_scale_list[it]
            trial_data: TrialData = self.run_trial(iteration=it,
                obj_scale=obj_scale, any_pose=self.any_pose,
                obj_shapenet_id=obj_shapenet_id)

            trial_result = trial_data.trial_result
            obj_shapenet_id = trial_data.obj_shapenet_id
            best_opt_idx = trial_data.aux_data['place_opt_idx']

            if trial_result == TrialResults.SUCCESS:
                num_success += 1

            log_info(f'Experiment: {self.experiment_type}')
            log_info(f'Trial result: {trial_result}')
            log_info(f'Shapenet id: {obj_shapenet_id}')
            log_str = f'Successes: {num_success} | Trials {it + 1} | ' \
                + f'Success Rate: {num_success / (it + 1):0.3f}'
            log_info(log_str)

            with open(self.global_summary_fname, 'a') as f:
                f.write(f'Trial number: {it}\n')
                f.write(f'Trial result: {trial_result}\n')
                f.write(f'Place teleport Success Rate: {num_success / (it + 1): 0.3f}\n')
                f.write(f'Shapenet id: {obj_shapenet_id}\n')
                f.write(f'Best Grasp idx: {best_opt_idx}\n')
                f.write('\n')


class EvaluateRackPlaceGrasp(EvaluateNetwork):
    def __init__(self, grasp_optimizer: OccNetOptimizer,
                 place_optimizer: OccNetOptimizer, seed: int,
                 shapenet_obj_dir: str, eval_save_dir: str,
                 demo_load_dir: str, pybullet_viz: bool = False,
                 test_obj_class: str = 'mug', num_trials: int = 200,
                 include_avoid_obj: bool = True, any_pose: bool = True):

        super().__init__(seed, shapenet_obj_dir, eval_save_dir,
            demo_load_dir, test_obj_class, pybullet_viz, num_trials,
            include_avoid_obj, any_pose)

        self.grasp_optimizer = grasp_optimizer
        self.place_optimizer = place_optimizer
        self.experiment_type = ExperimentTypes.RACK_PLACE_GRASP
        self.obj_sample_x_low_high = SimConstants.OBJ_SAMPLE_X_LOW_HIGH
        self.obj_sample_y_low_high = [-0.1, 0.1]
        # self.scale_low = 0.2
        # self.scale_high = 0.4

        # self.scale_high = 0.35
        # self.scale_low = 0.175
        self.obj_sample_x_low_high = [0.475, 0.5]
        # self.obj_sample_y_low_high = [-0.275, 0]
        self.obj_sample_y_low_high = [-0.2, 0]

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
        place_demo_fnames = [osp.join(self.demo_load_dir, fn) for fn in
            demo_fnames if 'place_demo' in fn]

        demo_shapenet_ids = set()
        for grasp_demo_fn in grasp_demo_fnames:
            print('Loading grasp demo from fname: %s' % grasp_demo_fn)
            grasp_data = np.load(grasp_demo_fn, allow_pickle=True)

            demo = DemoIO.process_grasp_data(grasp_data)

            self.grasp_optimizer.add_demo(demo)
            demo_shapenet_ids.add(demo.obj_shapenet_id)

        for place_demo_fn in place_demo_fnames:
            print('Loading place demo from fname: %s' % place_demo_fn)
            place_data = np.load(place_demo_fn, allow_pickle=True)

            demo = DemoIO.process_rack_place_data(place_data)

            self.place_optimizer.add_demo(demo)
            demo_shapenet_ids.add(demo.obj_shapenet_id)

        self.grasp_optimizer.process_demos()
        self.place_optimizer.process_demos()

        # -- Get table urdf -- #
        place_data = np.load(place_demo_fnames[0], allow_pickle=True)
        self.table_urdf = DemoIO.get_table_urdf(place_data)
        self.rack_pose = DemoIO.get_rack_pose(place_data)

        # -- Get test objects -- #
        self.test_object_ids = self._get_test_object_ids(demo_shapenet_ids)
        # self.test_object_ids = []
        # shapenet_id_list = [fn.split('_')[0]
        #     for fn in os.listdir(self.shapenet_obj_dir)]

        # for s_id in shapenet_id_list:
        #     valid = s_id not in demo_shapenet_ids \
        #         and s_id not in self.avoid_shapenet_ids

        #     if valid:
        #         self.test_object_ids.append(s_id)

    def configure_sim(self):
        """
        Run after demos are loaded
        """
        set_log_level('debug')
        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.left_pad_id,
            lateralFriction=3.0)
        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.right_pad_id,
            lateralFriction=3.0)

        self.robot.arm.reset(force_reset=True)

        self._set_up_cameras()

        # put table at right spot
        table_ori = euler2quat([0, 0, np.pi / 2])

        # Get raw table urdf
        # table_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table.urdf')
        table_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack.urdf')
        with open(table_urdf_fname, 'r', encoding='utf-8') as f:
            self.table_urdf = f.read()

        # this is the URDF that was used in the demos -- make sure we load an identical one
        tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack_tmp.urdf')
        open(tmp_urdf_fname, 'w').write(self.table_urdf)
        self.table_id = self.robot.pb_client.load_urdf(tmp_urdf_fname,
            SimConstants.TABLE_POS,
            table_ori,
            scaling=SimConstants.TABLE_SCALING)

    def run_trial(self, iteration: int = 0, obj_scale: float = -1,
                  any_pose: bool = True, thin_feature: bool = True,
                  grasp_viz: bool = False,
                  grasp_dist_thresh: float = 0.0025,
                  obj_shapenet_id: 'str | None' = None) -> TrialData:
        """
        Run trial where we try to grab object.

        Args:
            iteration (int, optional): What iteration the trial is. Defaults to 0.
            rand_mesh_scale (bool, optional): True to randomly scale mesh.
                Defaults to True.
            any_pose (bool, optional): True to use anypose function to pose mug.
                Defaults to True.
            thin_feature (bool, optional): True to treat object as thin feature
                in grasp post process. Defaults to True.
            grasp_viz (bool, optional): True to show image of grasp before trial
                runs. Only works when pybullet_viz is enabled. Defaults to False.
            grasp_dist_thresh (float, optional): Threshold to detect successful
                grasp. Defaults to 0.0025.
            obj_shapenet_id (str | None, optional): Object id to use.  If none,
                will randomly select id.

        Returns:
            TrialData: Class for storing relevant info about the trial
        """
        trial_data = TrialData()

        trial_data.aux_data = {
            'grasp_success': False,
            'place_success': False,
            'grasp_opt_idx': None,
            'place_opt_idx': None,
        }

        # -- Get and orient object -- #
        if obj_shapenet_id is None:
            obj_shapenet_id = random.sample(self.test_object_ids, 1)[0]
            log_info('Generate random obj id.')
        else:
            log_info('Using predefined obj id.')
        trial_data.obj_shapenet_id = obj_shapenet_id

        # Write at start so id is recorded regardless of any later bugs
        with open(self.shapenet_id_list_fname, 'a') as f:
            f.write(f'{trial_data.obj_shapenet_id}\n')

        # -- Home Robot -- #
        self.robot.arm.go_home(ignore_physics=True)
        self.robot.arm.move_ee_xyz([0, 0, 0.2])

        # -- load object -- #
        obj_id, o_cid, pos, ori = self._insert_object(obj_shapenet_id,
            obj_scale, any_pose)

        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        time.sleep(1.5)

        # -- Get object point cloud from cameras -- #
        target_obj_pcd_obs = self._get_pcd(obj_id)

        eval_iter_dir = osp.join(self.eval_save_dir, 'trial_%s' % str(iteration).zfill(3))
        util.safe_makedirs(eval_iter_dir)

        if target_obj_pcd_obs is None or target_obj_pcd_obs.shape[0] == 0:
            trial_data.trial_result = TrialResults.GET_PCD_FAILED
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        # -- Get grasp position -- #
        log_debug('Getting grasp position.')
        opt_viz_path = osp.join(eval_iter_dir, 'visualize')
        grasp_ee_pose_mats, best_grasp_idx = self.grasp_optimizer.optimize_transform_implicit(
            target_obj_pcd_obs, ee=True, viz_path=opt_viz_path)
        grasp_ee_pose = util.pose_stamped2list(util.pose_from_matrix(
            grasp_ee_pose_mats[best_grasp_idx]))
        trial_data.aux_data['grasp_opt_idx'] = best_grasp_idx

        # -- Post process grasp position -- #
        try:
            # When there are no nearby grasp points, this throws an index
            # error.  The try catch allows us to run more trials after the error.
            new_grasp_pt = post_process_grasp_point(
                grasp_ee_pose,
                target_obj_pcd_obs,
                thin_feature=thin_feature,
                grasp_viz=grasp_viz,
                grasp_dist_thresh=grasp_dist_thresh)
        except IndexError:
            trial_data.trial_result = TrialResults.POST_PROCESS_FAILED
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        grasp_ee_pose[:3] = new_grasp_pt

        # -- Create pose which offsets gripper from object -- #
        pregrasp_offset_tf = get_ee_offset(ee_pose=grasp_ee_pose)
        pre_grasp_ee_pose = util.pose_stamped2list(
            util.transform_pose(
                pose_source=util.list2pose_stamped(grasp_ee_pose),
                pose_transform=util.list2pose_stamped(pregrasp_offset_tf)))

        # -- Get place position -- #
        opt_viz_path = osp.join(eval_iter_dir, 'visualize')
        rack_pose_mats, best_place_idx = self.place_optimizer.optimize_transform_implicit(
            target_obj_pcd_obs, ee=False, viz_path=opt_viz_path)
        trial_data.aux_data['place_opt_idx'] = best_place_idx
        rack_relative_pose = util.transform_pose(
            util.pose_from_matrix(rack_pose_mats[best_place_idx]), util.list2pose_stamped(self.rack_pose))

        place_ee_pose = util.transform_pose(util.list2pose_stamped(grasp_ee_pose),
            pose_transform=rack_relative_pose)

        # preplace_offset_far_tf = util.list2pose_stamped(SimConstants.PREPLACE_OFFSET_FAR_TF)
        preplace_offset_far_tf = util.list2pose_stamped(SimConstants.PREPLACE_HORIZONTAL_OFFSET_TF)
        preplace_offset_close_tf = util.list2pose_stamped(SimConstants.PREPLACE_OFFSET_CLOSE_TF)
        place_far_pose = util.transform_pose(place_ee_pose, preplace_offset_far_tf)
        place_close_pose = util.transform_pose(place_ee_pose, preplace_offset_close_tf)

        place_ee_pose = util.pose_stamped2list(place_ee_pose)
        place_far_pose = util.pose_stamped2list(place_far_pose)
        place_close_pose = util.pose_stamped2list(place_close_pose)

        # -- Get ik -- #
        log_debug('Getting ik.')
        pre_grasp_jnt_pos = grasp_jnt_pos = None

        ik_status = []
        pre_grasp_jnt_pos, ik_res = self._compute_ik_cascade(pre_grasp_ee_pose)
        ik_status.append(ik_res)
        grasp_jnt_pos, ik_res = self._compute_ik_cascade(grasp_ee_pose)
        ik_status.append(ik_res)
        place_jnt_pose, ik_res = self._compute_ik_cascade(place_ee_pose)
        ik_status.append(ik_res)
        place_far_jnt_pose, ik_res = self._compute_ik_cascade(place_far_pose)
        ik_status.append(ik_res)
        place_close_jnt_pose, ik_res = self._compute_ik_cascade(place_close_pose)
        ik_status.append(ik_res)

        for ik_res in ik_status:
            if ik_res is not None:
                trial_data.trial_result = ik_res
                self.robot.pb_client.remove_body(obj_id)
                return trial_data

        # -- Prep for grasp -- #
        log_debug('Attempting grasp.')

        # turn OFF collisions between robot and object / table, and move to pre-grasp pose
        for i in range(p.getNumJoints(self.robot.arm.robot_id)):
            safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id,
                bodyUniqueIdB=self.table_id,
                linkIndexA=i,
                linkIndexB=-1,
                enableCollision=False,
                physicsClientId=self.robot.pb_client.get_client_id())

            safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id,
                bodyUniqueIdB=obj_id,
                linkIndexA=i,
                linkIndexB=-1,
                enableCollision=False,
                physicsClientId=self.robot.pb_client.get_client_id())

        home_jnt_pos = self.robot.arm.get_jpos()
        self.robot.arm.eetool.open()

        # -- Get grasp image -- #
        self.robot.pb_client.set_step_sim(True)
        self.robot.arm.set_jpos(grasp_jnt_pos, ignore_physics=True)
        self.robot.arm.eetool.close(ignore_physics=True)
        time.sleep(0.2)
        # grasp_rgb = self.robot.cam.get_images(get_rgb=True)[0]
        grasp_img_fname = osp.join(self.eval_grasp_imgs_dir,
            '%s_01pose.png' % str(iteration).zfill(3))
        # util.np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)
        self._take_image(grasp_img_fname)
        self.robot.arm.eetool.open(ignore_physics=True)
        self.robot.arm.go_home(ignore_physics=True)
        time.sleep(0.5)
        self.robot.pb_client.set_step_sim(False)

        # -- Plan grasp -- #
        # Get to pre grasp location
        plan1 = self.ik_helper.plan_joint_motion(home_jnt_pos, pre_grasp_jnt_pos)

        # Get to grasp location
        plan2 = self.ik_helper.plan_joint_motion(pre_grasp_jnt_pos, grasp_jnt_pos)

        # Return to home location (for checking if grasp was valid)
        plan3 = self.ik_helper.plan_joint_motion(grasp_jnt_pos, home_jnt_pos)

        if None in [plan1, plan2, plan3]:
            trial_data.trial_result = TrialResults.JOINT_PLAN_FAILED
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        # -- Move for grasp -- #
        self.robot.arm.eetool.open()
        # Go to pre grasp location (linearly away from grasp area)
        for jnt in plan1:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.025)
        self.robot.arm.set_jpos(plan1[-1], wait=False)

        # turn ON collisions between robot and object
        for i in range(p.getNumJoints(self.robot.arm.robot_id)):
            safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id,
                bodyUniqueIdB=obj_id,
                linkIndexA=i,
                linkIndexB=-1,
                enableCollision=True,
                physicsClientId=self.robot.pb_client.get_client_id())

        # Go to grasp location
        for jnt in plan2:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.04)
        self.robot.arm.set_jpos(plan2[-1], wait=False)

        time.sleep(0.8)

        obj_pos_before_grasp = p.getBasePositionAndOrientation(obj_id)[0]
        jnt_pos_before_grasp = self.robot.arm.get_jpos()

        soft_grasp_close(self.robot, RobotIDs.finger_joint_id, force=50)
        time.sleep(0.8)
        safeRemoveConstraint(o_cid)
        safeCollisionFilterPair(obj_id, self.table_id, -1, -1,
            enableCollision=False)
        time.sleep(0.8)

        grasp_img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_02grasp.png')
        self._take_image(grasp_img_fname)

        # Move to clearance location
        for jnt in plan3:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.025)
        self.robot.arm.set_jpos(plan3[-1], wait=False)
        time.sleep(1)

        grasp_success = object_is_still_grasped(self.robot,
            obj_id, RobotIDs.right_pad_id, RobotIDs.left_pad_id)
        time.sleep(0.5)

        # If the ee was intersecting the mug, original_grasp_success
        # would be true after the table disappears.  However, an
        # intersection is generally a false grasp When the ee is
        # opened again, a good grasp should fall down while an
        # intersecting grasp would stay in contact.

        # ^ This is not currently implemented cuz I don't wanna debug the mug moving
        # when its reset to that position.  If this error occurs, the place will
        # be false so its fine...

        # -- Take image of grasp at clearance height -- #
        grasp_img_fname = osp.join(self.eval_grasp_imgs_dir,
            '%s_03clearance.png' % str(iteration).zfill(3))
        self._take_image(grasp_img_fname)

        # Override drop test because it causes object to slip

        if grasp_success:
            trial_data.trial_result = TrialResults.GRASP_SUCCESS
            trial_data.aux_data['grasp_success'] = True
            # grasp_cid = constraint_grasp_close(self.robot, obj_id)


            # Get firm grasp on object, then move back to clearance

            self.robot.arm.eetool.open()
            p.resetBasePositionAndOrientation(obj_id, obj_pos_before_grasp, ori)
            o_cid = constraint_obj_world(obj_id, pos, ori) # Lock object in pose
            self.robot.arm.set_jpos(jnt_pos_before_grasp, ignore_physics=True)
            grasp_cid = constraint_grasp_close(self.robot, obj_id)
            soft_grasp_close(self.robot, RobotIDs.finger_joint_id, force=40)
            time.sleep(0.5)
            safeRemoveConstraint(o_cid)


            for jnt in plan3:
                self.robot.arm.set_jpos(jnt, wait=False)
                time.sleep(0.025)
            self.robot.arm.set_jpos(plan3[-1], wait=False)
            time.sleep(1)
        else:
            trial_data.trial_result = TrialResults.BAD_OPT_POS
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        # -- Set up for place -- #

        # NEW
        placement_link_id = 0

        # turn OFF collisions between object / table and object / rack, and move to pre-place pose
        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=False)
        safeCollisionFilterPair(obj_id, self.table_id, -1, placement_link_id, enableCollision=False)
        time.sleep(1.0)

        img_fname = osp.join(self.eval_grasp_imgs_dir,
            '%s_04clearance_place.png' % str(iteration).zfill(3))
        self._take_image(img_fname)

        # Go to far approach position
        plan1 = self.ik_helper.plan_joint_motion(home_jnt_pos, place_far_jnt_pose)

        # Go to close approach position
        plan2 = self.ik_helper.plan_joint_motion(place_far_jnt_pose, place_close_jnt_pose)

        # Go to final place position
        plan3 = self.ik_helper.plan_joint_motion(place_close_jnt_pose, place_jnt_pose)

        # Return to home position
        # plan4 = self.ik_helper.plan_joint_motion(place_jnt_pose, place_far_jnt_pose)
        # plan5 = self.ik_helper.plan_joint_motion(place_far_jnt_pose, home_jnt_pos)

        if None in [plan1, plan2, plan3]:
            trial_data.trial_result = TrialResults.JOINT_PLAN_FAILED
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        for jnt in plan1:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.04)
        self.robot.arm.set_jpos(plan1[-1], wait=False)

        for jnt in plan2:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.04)
        self.robot.arm.set_jpos(plan2[-1], wait=False)

        img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_05pre_place.png')
        self._take_image(img_fname)

        # NEW
        # turn ON collisions between object and rack, and open fingers
        # safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
        safeCollisionFilterPair(obj_id, self.table_id, -1, placement_link_id, enableCollision=True)

        time.sleep(0.5)

        for jnt in plan3:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.04)
        self.robot.arm.set_jpos(plan3[-1], wait=False)

        img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_06place.png')
        self._take_image(img_fname)

        # Open ee and remove rigid constraint
        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        constraint_grasp_open(grasp_cid)
        self.robot.arm.eetool.open()

        time.sleep(0.3)
        img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_07place_release.png')
        self._take_image(img_fname)

        # turn off collisions between robot and object
        for i in range(p.getNumJoints(self.robot.arm.robot_id)):
            safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id,
                bodyUniqueIdB=obj_id,
                linkIndexA=i,
                linkIndexB=-1,
                enableCollision=False,
                physicsClientId=self.robot.pb_client.get_client_id())

        self.robot.arm.go_home(ignore_physics=True)
        time.sleep(0.3)

        img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_08place_release_home.png')
        self._take_image(img_fname)

        # -- Check place was successful -- #
        # placement_link_id = 0
        # obj_surf_contacts = p.getContactPoints(obj_id, self.table_id, -1, placement_link_id)
        # touching_surf = len(obj_surf_contacts) > 0
        # place_success = touching_surf
        # observe and record outcome

        obj_surf_contacts = p.getContactPoints(obj_id, self.table_id, -1, placement_link_id)
        touching_surf = len(obj_surf_contacts) > 0
        obj_floor_contacts = p.getContactPoints(obj_id, self.robot.arm.floor_id, -1, -1)
        touching_floor = len(obj_floor_contacts) > 0
        place_success = touching_surf and not touching_floor

        if place_success:
            trial_data.trial_result = TrialResults.SUCCESS
            trial_data.aux_data['place_success'] = True

        self.robot.pb_client.remove_body(obj_id)
        return trial_data

    def run_experiment(self, rand_mesh_scale: bool = True, start_idx: int = 0):
        """
        Run experiment for {self.num_trials}
        """
        num_success = 0

        # obj_shapenet_id_list = ['2c1df84ec01cea4e525b133235812833-h'] + random.choices(self.test_object_ids, k=self.num_trials)
        obj_shapenet_id_list = random.choices(self.test_object_ids, k=self.num_trials)

        if self.test_obj_class == 'bottle':
            thin_feature = False
        else:
            thin_feature = True

        if rand_mesh_scale:
            obj_scale_list = np.random.random(self.num_trials).tolist()
        else:
            obj_scale_list = -1 * np.ones(self.num_trials)
            obj_scale_list = obj_scale_list.tolist()

        for it in range(start_idx, self.num_trials):
            obj_shapenet_id = obj_shapenet_id_list[it]
            obj_scale = obj_scale_list[it]
            trial_data: TrialData = self.run_trial(iteration=it,
                obj_scale=obj_scale, any_pose=self.any_pose,
                obj_shapenet_id=obj_shapenet_id, thin_feature=thin_feature)

            trial_result = trial_data.trial_result
            obj_shapenet_id = trial_data.obj_shapenet_id
            best_grasp_idx = trial_data.aux_data['grasp_opt_idx']
            best_place_idx = trial_data.aux_data['place_opt_idx']
            grasp_success = trial_data.aux_data['grasp_success']
            place_success = trial_data.aux_data['place_success']

            if trial_result == TrialResults.SUCCESS:
                num_success += 1

            log_info(f'Experiment: {self.experiment_type}')
            log_info(f'Trial result: {trial_result}')
            log_info(f'Shapenet id: {obj_shapenet_id}')
            log_info(f'Grasp Success: {grasp_success} | Place Success: {place_success}')
            log_str = f'Successes: {num_success} | Trials {it + 1} | ' \
                + f'Success Rate: {num_success / (it + 1):0.3f}'
            log_info(log_str)

            with open(self.global_summary_fname, 'a') as f:
                f.write(f'Trial number: {it}\n')
                f.write(f'Trial result: {trial_result}\n')
                f.write(f'Grasp Success Rate: {num_success / (it + 1): 0.3f}\n')
                f.write(f'Shapenet id: {obj_shapenet_id}\n')
                f.write(f'Best Grasp idx: {best_grasp_idx}\n')
                f.write(f'Best Place idx: {best_place_idx}\n')
                f.write('\n')


class EvaluateShelfPlaceGrasp(EvaluateNetwork):
    def __init__(self, grasp_optimizer: OccNetOptimizer,
                 place_optimizer: OccNetOptimizer, seed: int,
                 shapenet_obj_dir: str, eval_save_dir: str,
                 demo_load_dir: str, pybullet_viz: bool = False,
                 test_obj_class: str = 'mug', num_trials: int = 200,
                 include_avoid_obj: bool = True, any_pose: bool = True):

        super().__init__(seed, shapenet_obj_dir, eval_save_dir,
            demo_load_dir, test_obj_class, pybullet_viz, num_trials,
            include_avoid_obj, any_pose)

        self.grasp_optimizer = grasp_optimizer
        self.place_optimizer = place_optimizer
        self.experiment_type = ExperimentTypes.SHELF_PLACE_GRASP
        # self.obj_sample_x_low_high = SimConstants.OBJ_SAMPLE_X_LOW_HIGH
        self.obj_sample_x_low_high = [0.4, 0.45]
        # self.obj_sample_y_low_high = [-0.1, 0.1]
        self.obj_sample_y_low_high = [-0.3, -0.1]

        # # NEW
        # self.scale_low = 0.35
        # self.scale_high = 0.5
        # self.scale_default = 0.45

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
        place_demo_fnames = [osp.join(self.demo_load_dir, fn) for fn in
            demo_fnames if 'place_demo' in fn]

        demo_shapenet_ids = set()
        for grasp_demo_fn in grasp_demo_fnames:
            print('Loading grasp demo from fname: %s' % grasp_demo_fn)
            grasp_data = np.load(grasp_demo_fn, allow_pickle=True)

            demo = DemoIO.process_grasp_data(grasp_data)

            self.grasp_optimizer.add_demo(demo)
            demo_shapenet_ids.add(demo.obj_shapenet_id)

        for place_demo_fn in place_demo_fnames:
            print('Loading place demo from fname: %s' % place_demo_fn)
            place_data = np.load(place_demo_fn, allow_pickle=True)

            demo = DemoIO.process_shelf_place_data(place_data)

            self.place_optimizer.add_demo(demo)
            demo_shapenet_ids.add(demo.obj_shapenet_id)

        self.grasp_optimizer.process_demos()
        self.place_optimizer.process_demos()

        # -- Get table urdf -- #
        place_data = np.load(place_demo_fnames[0], allow_pickle=True)
        self.table_urdf = DemoIO.get_table_urdf(place_data)
        self.shelf_pose = DemoIO.get_shelf_pose(place_data)

        # -- Get test objects -- #
        self.test_object_ids = self._get_test_object_ids(demo_shapenet_ids)

    def configure_sim(self):
        """
        Run after demos are loaded
        """
        set_log_level('debug')
        # p.changeDynamics(self.robot.arm.robot_id, RobotIDs.left_pad_id,
        #     lateralFriction=1.0)
        # p.changeDynamics(self.robot.arm.robot_id, RobotIDs.right_pad_id,
        #     lateralFriction=1.0)

        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.left_pad_id,
            lateralFriction=3.0)
        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.right_pad_id,
            lateralFriction=3.0)

        self.robot.arm.reset(force_reset=True)

        self._set_up_cameras()

        # put table at right spot
        table_ori = euler2quat([0, 0, np.pi / 2])

        # Get raw table urdf
        # table_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_shelf.urdf')
        # with open(table_urdf_fname, 'r', encoding='utf-8') as f:
        #     self.table_urdf = f.read()

        # this is the URDF that was used in the demos -- make sure we load an identical one
        tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack_tmp.urdf')
        open(tmp_urdf_fname, 'w').write(self.table_urdf)
        self.table_id = self.robot.pb_client.load_urdf(tmp_urdf_fname,
            SimConstants.TABLE_POS,
            table_ori,
            scaling=SimConstants.TABLE_SCALING)

    def run_trial(self, iteration: int = 0, obj_scale: float = -1,
                  any_pose: bool = True, thin_feature: bool = True,
                  grasp_viz: bool = False,
                  grasp_dist_thresh: float = 0.0025,
                  obj_shapenet_id: 'str | None' = None) -> TrialData:
        """
        Run trial where we try to grab object.

        Args:
            iteration (int, optional): What iteration the trial is. Defaults to 0.
            rand_mesh_scale (bool, optional): True to randomly scale mesh.
                Defaults to True.
            any_pose (bool, optional): True to use anypose function to pose mug.
                Defaults to True.
            thin_feature (bool, optional): True to treat object as thin feature
                in grasp post process. Defaults to True.
            grasp_viz (bool, optional): True to show image of grasp before trial
                runs. Only works when pybullet_viz is enabled. Defaults to False.
            grasp_dist_thresh (float, optional): Threshold to detect successful
                grasp. Defaults to 0.0025.
            obj_shapenet_id (str | None, optional): Object id to use.  If none,
                will randomly select id.

        Returns:
            TrialData: Class for storing relevant info about the trial
        """
        trial_data = TrialData()

        trial_data.aux_data = {
            'grasp_success': False,
            'place_success': False,
            'grasp_opt_idx': None,
            'place_opt_idx': None,
        }

        # -- Get and orient object -- #
        if obj_shapenet_id is None:
            obj_shapenet_id = random.sample(self.test_object_ids, 1)[0]
            log_info('Generate random obj id.')
        else:
            log_info('Using predefined obj id.')
        trial_data.obj_shapenet_id = obj_shapenet_id

        # Write at start so id is recorded regardless of any later bugs
        with open(self.shapenet_id_list_fname, 'a') as f:
            f.write(f'{trial_data.obj_shapenet_id}\n')

        # -- Home Robot -- #
        self.robot.arm.go_home(ignore_physics=True)
        self.robot.arm.move_ee_xyz([0, 0, 0.2])

        # -- load object -- #
        obj_id, o_cid, pos, ori = self._insert_object(obj_shapenet_id,
            obj_scale, any_pose)

        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        time.sleep(1.5)

        # -- Get object point cloud from cameras -- #
        target_obj_pcd_obs = self._get_pcd(obj_id)

        eval_iter_dir = osp.join(self.eval_save_dir, 'trial_%s' % str(iteration).zfill(3))
        util.safe_makedirs(eval_iter_dir)

        if target_obj_pcd_obs is None or target_obj_pcd_obs.shape[0] == 0:
            trial_data.trial_result = TrialResults.GET_PCD_FAILED
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        # -- Get grasp position -- #
        log_debug('Getting grasp position.')
        opt_viz_path = osp.join(eval_iter_dir, 'visualize')
        grasp_ee_pose_mats, best_grasp_idx = self.grasp_optimizer.optimize_transform_implicit(
            target_obj_pcd_obs, ee=True, viz_path=opt_viz_path)
        grasp_ee_pose = util.pose_stamped2list(util.pose_from_matrix(
            grasp_ee_pose_mats[best_grasp_idx]))
        trial_data.aux_data['grasp_opt_idx'] = best_grasp_idx

        # -- Post process grasp position -- #
        try:
            # When there are no nearby grasp points, this throws an index
            # error.  The try catch allows us to run more trials after the error.
            new_grasp_pt = post_process_grasp_point(
                grasp_ee_pose,
                target_obj_pcd_obs,
                thin_feature=thin_feature,
                grasp_viz=grasp_viz,
                grasp_dist_thresh=grasp_dist_thresh)
        except IndexError:
            trial_data.trial_result = TrialResults.POST_PROCESS_FAILED
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        grasp_ee_pose[:3] = new_grasp_pt

        # -- Create pose which offsets gripper from object -- #
        pregrasp_offset_tf = get_ee_offset(ee_pose=grasp_ee_pose)
        pre_grasp_ee_pose = util.pose_stamped2list(
            util.transform_pose(
                pose_source=util.list2pose_stamped(grasp_ee_pose),
                pose_transform=util.list2pose_stamped(pregrasp_offset_tf)))

        post_grasp_offset_tf = util.list2pose_stamped(SimConstants.SHELF_GRASP_CLEARANCE_OFFSET)
        post_grasp_pos = util.pose_stamped2list(
            util.transform_pose(util.list2pose_stamped(grasp_ee_pose), post_grasp_offset_tf)
        )

        # -- Get place position -- #
        opt_viz_path = osp.join(eval_iter_dir, 'visualize')
        rack_pose_mats, best_place_idx = self.place_optimizer.optimize_transform_implicit(
            target_obj_pcd_obs, ee=False, viz_path=opt_viz_path)
        trial_data.aux_data['place_opt_idx'] = best_place_idx
        rack_relative_pose = util.transform_pose(
            util.pose_from_matrix(rack_pose_mats[best_place_idx]), util.list2pose_stamped(self.shelf_pose))

        place_ee_pose = util.transform_pose(util.list2pose_stamped(grasp_ee_pose),
            pose_transform=rack_relative_pose)

        # preplace_offset_far_tf = util.list2pose_stamped(SimConstants.PREPLACE_OFFSET_FAR_TF)
        # preplace_offset_far_tf = util.list2pose_stamped(SimConstants.PREPLACE_HORIZONTAL_OFFSET_TF)
        # preplace_offset_close_tf = util.list2pose_stamped(SimConstants.PREPLACE_OFFSET_CLOSE_TF)
        preplace_offset_tf = util.list2pose_stamped(SimConstants.SHELF_PREPLACE_OFFSET)
        preplace_pose = util.transform_pose(place_ee_pose, preplace_offset_tf)

        place_ee_pose = util.pose_stamped2list(place_ee_pose)
        preplace_pose = util.pose_stamped2list(preplace_pose)

        # -- Get ik -- #
        log_debug('Getting ik.')
        pre_grasp_jnt_pos = grasp_jnt_pos = None

        ik_status = []
        pre_grasp_jnt_pos, ik_res = self._compute_ik_cascade(pre_grasp_ee_pose)
        ik_status.append(ik_res)
        grasp_jnt_pos, ik_res = self._compute_ik_cascade(grasp_ee_pose)
        ik_status.append(ik_res)
        post_grasp_pos, ik_res = self._compute_ik_cascade(post_grasp_pos)
        ik_status.append(ik_res)
        place_jnt_pose, ik_res = self._compute_ik_cascade(place_ee_pose)
        ik_status.append(ik_res)
        preplace_jnt_pose, ik_res = self._compute_ik_cascade(preplace_pose)
        ik_status.append(ik_res)

        for ik_res in ik_status:
            if ik_res is not None:
                trial_data.trial_result = ik_res
                self.robot.pb_client.remove_body(obj_id)
                return trial_data

        # -- Prep for grasp -- #
        log_debug('Attempting grasp.')

        # turn OFF collisions between robot and object / table, and move to pre-grasp pose
        for i in range(p.getNumJoints(self.robot.arm.robot_id)):
            safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id,
                bodyUniqueIdB=self.table_id,
                linkIndexA=i,
                linkIndexB=-1,
                enableCollision=False,
                physicsClientId=self.robot.pb_client.get_client_id())

            safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id,
                bodyUniqueIdB=obj_id,
                linkIndexA=i,
                linkIndexB=-1,
                enableCollision=False,
                physicsClientId=self.robot.pb_client.get_client_id())

        home_jnt_pos = self.robot.arm.get_jpos()
        self.robot.arm.eetool.open()

        # -- Get grasp image -- #
        self.robot.pb_client.set_step_sim(True)
        self.robot.arm.set_jpos(grasp_jnt_pos, ignore_physics=True)
        self.robot.arm.eetool.close(ignore_physics=True)
        time.sleep(0.2)
        # grasp_rgb = self.robot.cam.get_images(get_rgb=True)[0]
        grasp_img_fname = osp.join(self.eval_grasp_imgs_dir,
            '%s_01pose.png' % str(iteration).zfill(3))
        # util.np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)
        self._take_image(grasp_img_fname)
        self.robot.arm.eetool.open(ignore_physics=True)
        self.robot.arm.go_home(ignore_physics=True)
        time.sleep(0.5)
        self.robot.pb_client.set_step_sim(False)

        # -- Plan grasp -- #
        # Get to pre grasp location
        plan1 = self.ik_helper.plan_joint_motion(home_jnt_pos, pre_grasp_jnt_pos)

        # Get to grasp location
        plan2 = self.ik_helper.plan_joint_motion(pre_grasp_jnt_pos, grasp_jnt_pos)

        # Move upwards to check if grasp was valid
        #TODO
        # plan3 = self.ik_helper.plan_joint_motion(grasp_jnt_pos, home_jnt_pos)
        plan3 = self.ik_helper.plan_joint_motion(grasp_jnt_pos, post_grasp_pos)

        # Return to home location (in preparation to place)
        # plan4 = self.ik_helper.plan_joint_motion(post_grasp_pos, home_jnt_pos)

        if None in [plan1, plan2, plan3]:
            trial_data.trial_result = TrialResults.JOINT_PLAN_FAILED
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        # -- Move for grasp -- #
        self.robot.arm.eetool.open()
        time.sleep(0.5)
        # Go to pre grasp location (linearly away from grasp area)
        for jnt in plan1:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.025)
        self.robot.arm.set_jpos(plan1[-1], wait=False)

        # # turn ON collisions between robot and object
        # for i in range(p.getNumJoints(self.robot.arm.robot_id)):
        #     safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id,
        #         bodyUniqueIdB=obj_id,
        #         linkIndexA=i,
        #         linkIndexB=-1,
        #         enableCollision=True,
        #         physicsClientId=self.robot.pb_client.get_client_id())

        # Go to grasp location
        for jnt in plan2:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.04)
        self.robot.arm.set_jpos(plan2[-1], wait=False)

        # turn ON collisions between robot and object
        for i in range(p.getNumJoints(self.robot.arm.robot_id)):
            safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id,
                bodyUniqueIdB=obj_id,
                linkIndexA=i,
                linkIndexB=-1,
                enableCollision=True,
                physicsClientId=self.robot.pb_client.get_client_id())

        time.sleep(0.8)

        obj_pos_before_grasp = p.getBasePositionAndOrientation(obj_id)[0]
        jnt_pos_before_grasp = self.robot.arm.get_jpos()

        # Testing with different close methods.
        # self.robot.arm.eetool.close()
        soft_grasp_close(self.robot, RobotIDs.finger_joint_id, force=40)
        time.sleep(0.8)

        grasp_img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_02grasp.png')
        self._take_image(grasp_img_fname)

        safeRemoveConstraint(o_cid)
        safeCollisionFilterPair(obj_id, self.table_id, -1, -1,
            enableCollision=False)
        time.sleep(0.8)

        # Move to clearance location
        for jnt in plan3:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.025)
        self.robot.arm.set_jpos(plan3[-1], wait=False)
        time.sleep(1)

        grasp_success = object_is_still_grasped(self.robot,
            obj_id, RobotIDs.right_pad_id, RobotIDs.left_pad_id)
        time.sleep(0.5)

        # If the ee was intersecting the mug, original_grasp_success
        # would be true after the table disappears.  However, an
        # intersection is generally a false grasp When the ee is
        # opened again, a good grasp should fall down while an
        # intersecting grasp would stay in contact.

        # ^ This is not currently implemented cuz I don't wanna debug the mug moving
        # when its reset to that position.  If this error occurs, the place will
        # be false so its fine...

        # -- Take image of grasp at clearance height -- #
        grasp_img_fname = osp.join(self.eval_grasp_imgs_dir,
            '%s_03clearance.png' % str(iteration).zfill(3))
        self._take_image(grasp_img_fname)

        if grasp_success:
            trial_data.trial_result = TrialResults.GRASP_SUCCESS
            trial_data.aux_data['grasp_success'] = True

            # Get firm grasp on object, then move back to clearance
            safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
            self.robot.arm.eetool.open()
            time.sleep(0.5)
            p.resetBasePositionAndOrientation(obj_id, obj_pos_before_grasp, ori)
            self.robot.arm.set_jpos(jnt_pos_before_grasp, ignore_physics=True)
            soft_grasp_close(self.robot, RobotIDs.finger_joint_id, force=40)
            grasp_cid = constraint_grasp_close(self.robot, obj_id)

            for jnt in plan3:
                self.robot.arm.set_jpos(jnt, wait=False)
                time.sleep(0.025)
            self.robot.arm.set_jpos(plan3[-1], wait=False)
            time.sleep(1)

            safeCollisionFilterPair(obj_id, self.table_id, -1, -1,
                enableCollision=False)

        else:
            trial_data.trial_result = TrialResults.BAD_OPT_POS
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        # for jnt in plan4:
        #     self.robot.arm.set_jpos(jnt, wait=False)
        #     time.sleep(0.025)
        # self.robot.arm.set_jpos(plan3[-1], wait=False)
        time.sleep(1)

        # -- Set up for place -- #
        placement_link_id = 0

        # turn OFF collisions between object / table and object / rack, and move to pre-place pose
        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=False)
        safeCollisionFilterPair(obj_id, self.table_id, -1, placement_link_id, enableCollision=False)
        time.sleep(1.0)

        img_fname = osp.join(self.eval_grasp_imgs_dir,
            '%s_04clearance_place.png' % str(iteration).zfill(3))
        self._take_image(img_fname)

        # Go to far approach position
        #TODO
        plan1 = self.ik_helper.plan_joint_motion(post_grasp_pos, home_jnt_pos)

        plan2 = self.ik_helper.plan_joint_motion(home_jnt_pos, preplace_jnt_pose)
        # plan2 = self.ik_helper.plan_joint_motion(post_grasp_pos, preplace_jnt_pose)

        # Go to close approach position
        plan3 = self.ik_helper.plan_joint_motion(preplace_jnt_pose, place_jnt_pose)

        # Return to home position
        # plan4 = self.ik_helper.plan_joint_motion(place_jnt_pose, place_far_jnt_pose)
        # plan5 = self.ik_helper.plan_joint_motion(place_far_jnt_pose, home_jnt_pos)

        if None in [plan1, plan2, plan3]:
            trial_data.trial_result = TrialResults.JOINT_PLAN_FAILED
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        for jnt in plan1:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.04)
        self.robot.arm.set_jpos(plan1[-1], wait=False)

        for jnt in plan2:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.04)
        self.robot.arm.set_jpos(plan2[-1], wait=False)

        for jnt in plan3:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.04)
        self.robot.arm.set_jpos(plan3[-1], wait=False)

        img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_05pre_place.png')
        self._take_image(img_fname)

        # turn ON collisions between object and rack, and open fingers
        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
        safeCollisionFilterPair(obj_id, self.table_id, -1, placement_link_id, enableCollision=True)
        time.sleep(0.5)

        img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_06place.png')
        self._take_image(img_fname)

        # p.changeDynamics(self.robot.arm.robot_id, RobotIDs.left_pad_id,
        #     lateralFriction=4.0)
        # p.changeDynamics(self.robot.arm.robot_id, RobotIDs.right_pad_id,
        #     lateralFriction=4.0)

        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        self.robot.arm.eetool.open()
        constraint_grasp_open(grasp_cid)

        time.sleep(0.3)
        img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_07place_release.png')
        self._take_image(img_fname)

        self.robot.arm.go_home(ignore_physics=True)
        time.sleep(0.3)

        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=False)
        img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_08place_release_home.png')
        self._take_image(img_fname)

        # -- Check place was successful -- #

        placement_link_id = 0
        obj_surf_contacts = p.getContactPoints(obj_id, self.table_id, -1, placement_link_id)
        touching_surf = len(obj_surf_contacts) > 0
        place_success = touching_surf
        if place_success:
            trial_data.trial_result = TrialResults.SUCCESS
            trial_data.aux_data['place_success'] = True

        self.robot.pb_client.remove_body(obj_id)
        return trial_data

    def run_experiment(self, rand_mesh_scale: bool = True, start_idx: int = 0):
        """
        Run experiment for {self.num_trials}
        """
        num_success = 0

        obj_shapenet_id_list = random.choices(self.test_object_ids, k=self.num_trials)

        # # TODO
        # obj_shapenet_id_list = ['2c1df84ec01cea4e525b133235812833-h']

        if self.test_obj_class == 'bottle':
            thin_feature = False
        else:
            thin_feature = True

        if rand_mesh_scale:
            obj_scale_list = np.random.random(self.num_trials).tolist()
        else:
            obj_scale_list = -1 * np.ones(self.num_trials)
            obj_scale_list = obj_scale_list.tolist()

        for it in range(start_idx, self.num_trials):
            obj_shapenet_id = obj_shapenet_id_list[it]
            obj_scale = obj_scale_list[it]
            trial_data: TrialData = self.run_trial(iteration=it,
                obj_scale=obj_scale, any_pose=self.any_pose,
                obj_shapenet_id=obj_shapenet_id, thin_feature=thin_feature)

            trial_result = trial_data.trial_result
            obj_shapenet_id = trial_data.obj_shapenet_id
            best_grasp_idx = trial_data.aux_data['grasp_opt_idx']
            best_place_idx = trial_data.aux_data['place_opt_idx']
            grasp_success = trial_data.aux_data['grasp_success']
            place_success = trial_data.aux_data['place_success']

            if trial_result == TrialResults.SUCCESS:
                num_success += 1

            log_info(f'Experiment: {self.experiment_type}')
            log_info(f'Trial result: {trial_result}')
            log_info(f'Shapenet id: {obj_shapenet_id}')
            log_info(f'Grasp Success: {grasp_success} | Place Success: {place_success}')
            log_str = f'Successes: {num_success} | Trials {it + 1} | ' \
                + f'Success Rate: {num_success / (it + 1):0.3f}'
            log_info(log_str)

            with open(self.global_summary_fname, 'a') as f:
                f.write(f'Trial number: {it}\n')
                f.write(f'Trial result: {trial_result}\n')
                f.write(f'Grasp Success Rate: {num_success / (it + 1): 0.3f}\n')
                f.write(f'Shapenet id: {obj_shapenet_id}\n')
                f.write(f'Best Grasp idx: {best_grasp_idx}\n')
                f.write(f'Best Place idx: {best_place_idx}\n')
                f.write('\n')


class EvaluateNetworkSetup():
    """
    Set up experiment from config file
    """
    def __init__(self):
        self.config_dir = osp.join(path_util.get_ndf_eval(), 'eval_configs')
        self.config_dict = None
        self.seed = None

    def set_up_network(self, fname: str) -> EvaluateNetwork:
        config_path = osp.join(self.config_dir, fname)
        with open(config_path, "r") as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.config_dict = config_dict
        setup_dict = self.config_dict['setup_args']
        self.seed = setup_dict['seed']

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        print(config_dict)

        evaluator_type: str = setup_dict['evaluator_type']

        if evaluator_type == 'GRASP':
            return self._grasp_setup()
        elif evaluator_type == 'RACK_PLACE_TELEPORT':
            return self._rack_place_teleport_setup()
        elif evaluator_type == 'SHELF_PLACE_TELEPORT':
            return self._shelf_place_teleport_setup()
        elif evaluator_type == 'RACK_PLACE_GRASP':
            return self._rack_place_grasp_setup()
        elif evaluator_type == 'SHELF_PLACE_GRASP':
            return self._shelf_place_grasp_setup()
        else:
            raise ValueError('Invalid evaluator type.')

    def _grasp_setup(self) -> EvaluateNetwork:
        setup_config = self.config_dict['setup_args']
        obj_class = self.config_dict['evaluator']['test_obj_class']

        model = self._create_model(self.config_dict['model'])
        gripper_query_pts = self._create_query_pts(self.config_dict['gripper_query_pts'])
        eval_save_dir = self._create_eval_dir(setup_config)

        evaluator_config = self.config_dict['evaluator']
        shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
            obj_class + '_centered_obj_normalized')

        demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos',
            setup_config['demo_exp'])

        grasp_optimizer = self._create_optimizer(self.config_dict['grasp_optimizer'],
            model, gripper_query_pts, eval_save_dir)

        experiment = EvaluateGrasp(grasp_optimizer=grasp_optimizer,
            seed=self.seed, shapenet_obj_dir=shapenet_obj_dir,
            eval_save_dir=eval_save_dir, demo_load_dir=demo_load_dir,
            **evaluator_config)

        return experiment

    def _rack_place_teleport_setup(self) -> EvaluateNetwork:
        setup_config = self.config_dict['setup_args']
        obj_class = self.config_dict['evaluator']['test_obj_class']

        model = self._create_model(self.config_dict['model'])
        rack_query_pts = self._create_query_pts(self.config_dict['rack_query_pts'])
        eval_save_dir = self._create_eval_dir(setup_config)

        evaluator_config = self.config_dict['evaluator']
        shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
            obj_class + '_centered_obj_normalized')

        demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos',
            setup_config['demo_exp'])

        place_optimizer = self._create_optimizer(self.config_dict['place_optimizer'],
            model, rack_query_pts, eval_save_dir)

        experiment = EvaluateRackPlaceTeleport(place_optimizer=place_optimizer,
            seed=self.seed, shapenet_obj_dir=shapenet_obj_dir,
            eval_save_dir=eval_save_dir, demo_load_dir=demo_load_dir,
            **evaluator_config)

        return experiment

    def _shelf_place_teleport_setup(self) -> EvaluateNetwork:
        setup_config = self.config_dict['setup_args']
        obj_class = self.config_dict['evaluator']['test_obj_class']

        model = self._create_model(self.config_dict['model'])
        shelf_query_pts = self._create_query_pts(self.config_dict['shelf_query_pts'])
        eval_save_dir = self._create_eval_dir(setup_config)

        evaluator_config = self.config_dict['evaluator']
        shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
            obj_class + '_centered_obj_normalized')

        demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos',
            setup_config['demo_exp'])

        place_optimizer = self._create_optimizer(self.config_dict['place_optimizer'],
            model, shelf_query_pts, eval_save_dir)

        experiment = EvaluateShelfPlaceTeleport(place_optimizer=place_optimizer,
            seed=self.seed, shapenet_obj_dir=shapenet_obj_dir,
            eval_save_dir=eval_save_dir, demo_load_dir=demo_load_dir,
            **evaluator_config)

        return experiment

    def _rack_place_grasp_setup(self) -> EvaluateNetwork:
        setup_config = self.config_dict['setup_args']
        obj_class = self.config_dict['evaluator']['test_obj_class']

        model = self._create_model(self.config_dict['model'])
        gripper_query_pts = self._create_query_pts(self.config_dict['gripper_query_pts'])
        rack_query_pts = self._create_query_pts(self.config_dict['rack_query_pts'])
        eval_save_dir = self._create_eval_dir(setup_config)

        evaluator_config = self.config_dict['evaluator']
        shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
            obj_class + '_centered_obj_normalized')

        demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos',
            setup_config['demo_exp'])

        grasp_optimizer = self._create_optimizer(self.config_dict['grasp_optimizer'],
            model, gripper_query_pts, eval_save_dir)
        place_optimizer = self._create_optimizer(self.config_dict['place_optimizer'],
            model, rack_query_pts, eval_save_dir)

        experiment = EvaluateRackPlaceGrasp(grasp_optimizer=grasp_optimizer,
            place_optimizer=place_optimizer,
            seed=self.seed, shapenet_obj_dir=shapenet_obj_dir,
            eval_save_dir=eval_save_dir, demo_load_dir=demo_load_dir,
            **evaluator_config)

        return experiment

    def _shelf_place_grasp_setup(self) -> EvaluateNetwork:
        setup_config = self.config_dict['setup_args']
        obj_class = self.config_dict['evaluator']['test_obj_class']

        model = self._create_model(self.config_dict['model'])
        gripper_query_pts = self._create_query_pts(self.config_dict['gripper_query_pts'])
        shelf_query_pts = self._create_query_pts(self.config_dict['shelf_query_pts'])
        eval_save_dir = self._create_eval_dir(setup_config)

        evaluator_config = self.config_dict['evaluator']
        shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
            obj_class + '_centered_obj_normalized')

        # TODO Mod
        # shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
        #     obj_class + '_handle_centered_obj_normalized_res1')

        demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos',
            setup_config['demo_exp'])

        grasp_optimizer = self._create_optimizer(self.config_dict['grasp_optimizer'],
            model, gripper_query_pts, eval_save_dir)
        place_optimizer = self._create_optimizer(self.config_dict['place_optimizer'],
            model, shelf_query_pts, eval_save_dir)

        experiment = EvaluateShelfPlaceGrasp(grasp_optimizer=grasp_optimizer,
            place_optimizer=place_optimizer,
            seed=self.seed, shapenet_obj_dir=shapenet_obj_dir,
            eval_save_dir=eval_save_dir, demo_load_dir=demo_load_dir,
            **evaluator_config)

        return experiment

    def _create_model(self, model_config) -> torch.nn.Module:
        """
        Create torch model from given configs

        Returns:
            torch.nn.Module: Either ConvOccNetwork or VNNOccNet
        """
        model_type = model_config['type']
        model_args = model_config['args']
        model_checkpoint = osp.join(path_util.get_ndf_model_weights(),
                                    model_config['checkpoint'])

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

    def _create_optimizer(self, optimizer_config: dict, model: torch.nn.Module,
            query_pts: np.ndarray, eval_save_dir=None) -> OccNetOptimizer:
        """
        Create OccNetOptimizer from given config

        Args:
            model (torch.nn.Module): Model to use in the optimizer
            query_pts (np.ndarray): Query points to use in optimizer

        Returns:
            OccNetOptimizer: Optimizer to find best grasp position
        """
        optimizer_config = optimizer_config['args']
        if eval_save_dir is not None:
            opt_viz_path = osp.join(eval_save_dir, 'visualization')
        else:
            opt_viz_path = 'visualization'

        optimizer = OccNetOptimizer(model, query_pts, viz_path=opt_viz_path,
            **optimizer_config)
        return optimizer

    def _create_query_pts(self, query_pts_config: dict) -> np.ndarray:
        """
        Create query points from given config

        Args:
            query_pts_config(dict): Configs loaded from yaml file.

        Returns:
            np.ndarray: Query point as ndarray
        """

        query_pts_type = query_pts_config['type']
        query_pts_args = query_pts_config['args']

        assert query_pts_type in QueryPointTypes, 'Invalid query point type'

        if query_pts_type == 'SPHERE':
            query_pts = QueryPoints.generate_sphere(**query_pts_args)
        elif query_pts_type == 'RECT':
            query_pts = QueryPoints.generate_rect(**query_pts_args)
        elif query_pts_type == 'ARM':
            query_pts = QueryPoints.generate_rack_arm(**query_pts_args)
        elif query_pts_type == 'SHELF':
            query_pts = QueryPoints.generate_shelf(**query_pts_args)

        return query_pts

    def _create_eval_dir(self, setup_config: dict) -> str:
        """
        Create eval save dir as concatenation of current time
        and 'exp_desc'.

        Args:
            exp_desc (str, optional): Description of experiment. Defaults to ''.

        Returns:
            str: eval_save_dir.  Gives access to eval save directory
        """
        if 'exp_dir_suffix' in setup_config:
            exp_desc = setup_config['exp_dir_suffix']
        else:
            exp_desc = ''
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

        config_fname_yml = osp.join(eval_save_dir, 'config.yml')
        config_fname_txt = osp.join(eval_save_dir, 'config.txt')
        with open(config_fname_yml, 'w') as f:
            yaml.dump(self.config_dict, f)

        with open(config_fname_txt, 'w') as f:
            yaml.dump(self.config_dict, f)

        return eval_save_dir

    def _get_demo_load_dir(self,
        demo_exp: str='mug/grasp_rim_hang_handle_gaussian_precise_w_shelf') -> str:
        """
        Get directory of demos

        Args:
            obj_class (str, optional): Object class. Defaults to 'mug'.
            demo_exp (str, optional): Demo experiment name. Defaults to
                'grasp_rim_hang_handle_gaussian_precise_w_shelf'.

        Returns:
            str: Path to demo load dir
        """
        # demo_load_dir = osp.join(path_util.get_ndf_data(),
        #                          'demos', obj_class, demo_exp)

        demo_load_dir = osp.join(path_util.get_ndf_data(),
                                 'demos', demo_exp)

        return demo_load_dir


if __name__ == '__main__':
    # config_fname = 'debug_config.yml'
    # config_fname = 'debug_config_ndf.yml'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fname', type=str, default='debug_config.yml',
                        help='Filename of experiment config yml')

    args = parser.parse_args()
    config_fname = args.config_fname

    setup = EvaluateNetworkSetup()
    experiment = setup.set_up_network(config_fname)
    experiment.load_demos()
    experiment.configure_sim()
    experiment.run_experiment()
    # experiment.run_experiment(start_idx=10)

    # setup = EvaluateNetworkSetup()
    # setup.load_config(config_fname)
    # model = setup.create_model()
    # gripper_query_pts = setup.create_gripper_query_pts()
    # rack_query_pts = setup.create_rack_query_pts()

    # shapenet_obj_dir = setup.get_shapenet_obj_dir()
    # eval_save_dir = setup.create_eval_dir()
    # demo_load_dir = setup.get_demo_load_dir(obj_class='mug')

    # grasp_optimizer = setup.create_grasp_optimizer(model, gripper_query_pts, eval_save_dir=eval_save_dir)

    # evaluator_args = setup.get_evaluator_args()

    # experiment = EvaluateGrasp(grasp_optimizer=grasp_optimizer,
    #     seed=setup.get_seed(),
    #     shapenet_obj_dir=shapenet_obj_dir, eval_save_dir=eval_save_dir,
    #     demo_load_dir=demo_load_dir, **evaluator_args)

    # experiment.load_demos()
    # experiment.configure_sim()
    # experiment.run_experiment()
