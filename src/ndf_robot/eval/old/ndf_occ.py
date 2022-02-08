import os, os.path as osp
import torch
import numpy as np
import trimesh
import random
import copy
import plotly.graph_objects as go

from ndf_robot.utils import torch_util, trimesh_util
from ndf_robot.utils.plotly_save import plot3d


class NDFOcc():
    def __init__(self, model, model_type='pointnet'):
        self.model = model
        self.model_type = model_type

        self.loss_fn = torch.nn.L1Loss()
        if torch.cuda.is_available():
            self.dev = torch.device('cuda:0')
        else:
            self.dev = torch.device('cpu')
        print('device %s' % self.dev)

        self.model = self.model.to(self.dev)
        self.model.eval()
    
    def test_point(self, point_cloud, query_point):
        torch_pcd = torch.from_numpy(point_cloud).float().to(self.dev)
        torch_query_pnt = torch.from_numpy(query_point).float().to(self.dev)
        input_dict = {
            'point_cloud': torch_pcd,
            'coords': torch_query_pnt,
        }
        print("Input dict coord size: ", input_dict['coords'].shape)
        print("Input dict point cloud size", input_dict['point_cloud'].shape)
        with torch.no_grad():
            out_dict = self.model(input_dict) 
            return out_dict['occ']