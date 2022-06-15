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
import os.path as osp
import yaml
import random
from datetime import datetime

import torch

import pybullet as p

from airobot import Robot
from airobot.utils import common
from airobot.utils.common import euler2quat
from ndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from ndf_robot.config.default_obj_cfg import get_obj_cfg_defaults

from ndf_robot.utils import path_util, util

import ndf_robot.model.vnn_occupancy_net.vnn_occupancy_net_pointnet_dgcnn \
    as vnn_occupancy_network
import ndf_robot.model.conv_occupancy_net.conv_occupancy_net \
    as conv_occupancy_network

from ndf_robot.opt.optimizer import OccNetOptimizer
from ndf_robot.utils.franka_ik import FrankaIK


ModelTypes = {
    'CONV_OCC',
    'VNN_NDF',
}

QueryPointTypes = {
    'SPHERE'
}


class EvaluateGrasp():
    def __init__(self, optimizer: OccNetOptimizer, seed: int, pybullet_viz: False):
        self.optimizer = optimizer
        self.seed = seed

        self.robot = Robot('franka',
                           pb_cfg={'gui': pybullet_viz},
                           arm_cfg={'self_collision': False, 'seed': seed})
        self.ik_helper = FrankaIK(gui=False)

        # Get default config
        cfg = get_eval_cfg_defaults()
        cfg.freeze()

        obj_cfg = get_obj_cfg_defaults()




        # self.cfg = get_eval_cfg_defaults()






class EvaluateGraspParser():
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

        assert model_type in ModelTypes, 'Invalid model type'

        if model_type == 'CONV_OCC':
            model = conv_occupancy_network.ConvolutionalOccupancyNetwork(
                **model_args
            )
        elif model_type == 'VNN_NDF':
            model = vnn_occupancy_network.VNNOccNet(
                **model_args
            )

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

    parser = EvaluateGraspParser()
    parser.load_config(config_fname)
    model = parser.create_model()
    query_pts = parser.create_query_pts()
    optimizer = parser.create_optimizer(model, query_pts)

    print(optimizer)
