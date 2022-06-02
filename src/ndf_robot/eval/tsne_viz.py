"""
Generate TSNE visualization of mug with given model weights
"""
import os.path as osp

import torch
import numpy as np
import trimesh
from sklearn.manifold import TSNE
from scipy.spatial.transform import Rotation
import plotly.express as px

import ndf_robot.model.conv_occupancy_net.conv_occupancy_net as conv_occupancy_network
from ndf_robot.utils import path_util, torch3d_util, torch_util
from ndf_robot.utils.plotly_save import plot3d


class TSNEViz:
    """
    Visualize the latent activations of a model on any given
    object
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.dev = torch.device('cuda:0') if torch.cuda.is_available() \
            else torch.device('cpu')

    def viz_object(self, pcd: np.ndarray, query_pts: np.ndarray,
                   output_fn: str='tnse_viz.html', n_components=1,
                   rand_rotate: bool=False):
        """
        Generate TSNE plot of object given model and query points
        Saves plot as html to output_fn 

        Args:
            object_fn (str): filename of object mesh to load
            query_pts (np.ndarray): n x 3 array of points to sample from 
            output_fn (str): filename of output html to save to (must include html)

        UPDATE
        """
        model_input = {}
        query_pts_torch = torch.from_numpy(query_pts).float().to(self.dev)
        pcd_torch = torch.from_numpy(pcd).float().to(self.dev)

        if rand_rotate:
            random_transform = torch.tensor(TSNEViz.__random_rot_transform()).float().to(self.dev)
            query_pts_torch = torch_util.transform_pcd_torch(query_pts_torch, 
                random_transform).float()
            pcd_torch = torch_util.transform_pcd_torch(pcd_torch,
                random_transform).float()
            
        model_input['coords'] = query_pts_torch[None, :, :]
        model_input['point_cloud'] = pcd_torch[None, :, :]

        # n_query_pts = query_pts.shape[0]

        # model_input['coords'] = query_pts[None, :, :][:, [1, 0, 2]]
        # model_input['point_cloud'] = pcd[None, :, :]

        print('input: ', model_input['point_cloud'].shape)
        print('query: ', model_input['coords'].shape)

        latent_torch = self.model.extract_latent(model_input).detach()
        act_torch = self.model.forward_latent(latent_torch, model_input['coords']).detach()
        act = act_torch.squeeze().cpu().numpy()
        print('act_np', act.shape)

        tsne = TSNE(n_components)
        tsne_result = tsne.fit_transform(act)

        # plot3d([pcd, query_pts], ['blue', 'red'], 'tsne_plot.html',
        #        auto_scene=True)
        
        # fig = px.scatter(x=tsne_result[:, 0], y=tsne_result[:, 1], 
        #     labels={'x': 'tsne1', 'y': 'tsne2'})

        query_pts_np = query_pts_torch.cpu().numpy()
        
        fig = px.scatter_3d(x=query_pts_np[:, 0], y=query_pts_np[:, 1], z=query_pts_np[:, 2], 
            color=tsne_result[:, 0])
        
        fig.write_html(output_fn)
    
    @staticmethod
    def __random_quaternions(n: int):
        """
        Generate random quaternions representing rotations,
        i.e. versors with nonnegative real part.

        Modified from random_quaternions in util.torch3d_util.py

        Args:
            n: Number of quaternions in a batch to return.
            dtype: Type to return.
            device: Desired device of returned tensor. Default:
                uses the current device for the default tensor type.

        Returns:
            Quaternions as tensor of shape (N, 4).
        """
        o = torch.randn((n, 4))
        s = (o * o).sum(1)
        o = o / torch3d_util._copysign(torch.sqrt(s), o[:, 0])[:, None]
        return o

    @staticmethod
    def __random_rot_transform():
        """
        Generate a random rotation transform

        Args:
            translate (bool, optional): True to include translation in 
                transform. Defaults to False.

        Raises:
            NotImplementedError: translation not done yet

        Returns: 
            Transform with random rotation
        """
        rand_quat = TSNEViz.__random_quaternions(1)

        rand_rot = Rotation.from_quat(rand_quat)
        rand_rot = rand_rot.as_matrix()

        transform = np.eye(4)
        transform[:3, :3] = rand_rot

        return transform
    

if __name__ == '__main__':

    # CONSTANTS #
    # object_fn = osp.join(path_util.get_ndf_demo_obj_descriptions(),
    #     'mug_centered_obj_normalized/586e67c53f181dc22adf8abaa25e0215/models/model_normalized.obj')
    # object_fn= osp.join(path_util.get_ndf_demo_obj_descriptions(),
    #     'mug_centered_obj_normalized/edaf960fb6afdadc4cebc4b5998de5d0/models/model_normalized.obj')
    # object_fn = osp.join(path_util.get_ndf_demo_obj_descriptions(),
    #     'mug_centered_obj_normalized/e984fd7e97c2be347eaeab1f0c9120b7/models/model_normalized.obj')
    object_fn = osp.join(path_util.get_ndf_demo_obj_descriptions(),
        'mug_centered_obj_normalized/ec846432f3ebedf0a6f32a8797e3b9e9//models/model_normalized.obj')
    output_fn = 'tsne_viz.html'
    # output_fn = 'tsne_viz_latent_32.html'

    # LOAD MODEL #
    # model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=32,
    #     model_type='pointnet', return_features=True, sigmoid=True).cuda()

    model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=4,
        model_type='pointnet', return_features=True, sigmoid=True).cuda()

    # model_path = osp.join(path_util.get_ndf_model_weights(), 
    #     'ndf_vnn/conv_occ_latent_transfer_rand_coords_margin_no_neg_margin_1/checkpoints/model_epoch_0011_iter_143000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 
    #     'ndf_vnn/conv_occ_latent_log_2/checkpoints/model_epoch_0000_iter_001000.pth')

    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_latent_adaptive_2/checkpoints/model_epoch_0009_iter_099000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_latent_4_0/checkpoints/model_epoch_0010_iter_130000.pth')
    model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_latent_dim4_rotated_triplet_0/checkpoints/model_epoch_0000_iter_002000.pth')
    model.load_state_dict(torch.load(model_path))

    # SET QUERY POINTS #
    mesh = trimesh.load(object_fn, process=False)
    # pcd = trimesh.sample.volume_mesh(mesh, 5000) # pcd of object
    pcd = mesh.sample(5000)
    query_pts = mesh.sample(500)  # Set to also sample within body

    # RUN PLOTTER #
    tsne_plotter = TSNEViz(model)
    tsne_plotter.viz_object(pcd, query_pts, output_fn, rand_rotate=True)
    # tsne_plotter.viz_object(pcd, query_pts, output_fn, rand_rotate=False)
