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

from ndf_robot.utils import path_util

import ndf_robot.model.vnn_occupancy_net.vnn_occupancy_net_pointnet_dgcnn \
    as vnn_occupancy_network
import ndf_robot.model.conv_occupancy_net.conv_occupancy_net \
    as conv_occupancy_network

from ndf_robot.opt.optimizer import OccNetOptimizer


class ModelType(Enum):
    VNN_NDF = 1
    CONV_OCC = 2


class QueryPtType(Enum):
    SPHERE = 1


class EvaluateGrasp():
    def __init__(self, model_type: ModelType, model_args: 'dict[str, any]',
        optimizer_args: 'dict[str, any]', query_pt_type: QueryPtType,
        query_pt_args: 'dict[str, any]', seed: int=0):

        # Set model
        self.model_type = model_type
        if self.model_type == ModelType.VNN_NDF:
            self.model = vnn_occupancy_network.VNNOccNet(**model_args)
        elif self.model_type == ModelType.CONV_OCC:
            self.model = conv_occupancy_network.ConvolutionalOccupancyNetwork(
                **model_args)

        # Set query points
        if query_pt_type == QueryPtType.SPHERE:
            self.query_pts = QueryPoints.generate_sphere(**query_pt_args)

        # Set optimizer
        self.optimizer = OccNetOptimizer(self.model, self.query_pts,
            **optimizer_args)


class EvaluateGraspParser():
    """
    Set up experiment from config file

    File format is yaml with

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

    """
    def __init__(self):
        self.config_dir = osp.join(path_util.get_ndf_eval(), 'eval_configs')

    def load_from_dicts(model_args, optimizer_args, query_pt_args)


class QueryPoints():
    @staticmethod
    def generate_sphere(n_pts, radius=0.05):
        # http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        u = 2 * np.random.rand(n_pts, 1) - 1
        phi = 2 * np.pi * np.random.rand(n_pts, 1)
        r = radius * (np.random.rand(n_pts, 1)**(1 / 3.))
        x = r * np.cos(phi) * (1 - u**2)**0.5
        y = r * np.sin(phi) * (1 - u**2)**0.5
        z = r * u

        sphere_points = np.hstack((x, y, z))
        return sphere_points
