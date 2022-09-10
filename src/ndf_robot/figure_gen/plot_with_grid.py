import os.path as osp

import numpy as np
import torch
from torch.nn import functional as F
import trimesh

from scipy.spatial.transform import Rotation as R
import plotly.express as px

from ndf_robot.utils import path_util, util
from ndf_robot.utils.plotly_save import multiplot

import ndf_robot.model.vnn_occupancy_net.vnn_occupancy_net_pointnet_dgcnn \
    as vnn_occupancy_network
import ndf_robot.model.conv_occupancy_net.conv_occupancy_net \
    as conv_occupancy_network

import plotly.express as px
import plotly.graph_objects as go

def add_plane(fig, normal_ax: str, x_extents: tuple, y_extents: tuple, z_extents: tuple,
    axis_loc: float, color: np.ndarray):
    if normal_ax == 'z':
        x = np.linspace(x_extents[0], x_extents[1], 20)
        y = np.linspace(y_extents[0], y_extents[1], 20)
        z = axis_loc * np.ones((20, 20))
    if normal_ax == 'x':
        x = np.linspace(axis_loc, axis_loc, 20)
        y = np.linspace(y_extents[0], y_extents[1], 20)
        z = np.repeat(np.linspace(z_extents[0], z_extents[1], 20).reshape(1, 20), 20, axis=0)

    if normal_ax == 'y':
        x = np.linspace(x_extents[0], x_extents[1], 20)
        y = np.linspace(axis_loc, axis_loc, 20)
        z = np.repeat(np.linspace(z_extents[0], z_extents[1], 20).reshape(20, 1), 20, axis=1)

    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=color,  showscale=False))

    return fig

if __name__ == '__main__':

    # seed = 0
    seed = 6
    # seed = 2

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    use_conv = True
    # use_conv = False
    n_samples = 1000

    if torch.cuda.is_available():
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    # see the demo object descriptions folder for other object models you can try
    # obj_model = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/28f1e7bc572a633cb9946438ed40eeb9/models/model_normalized.obj')
    obj_model = osp.join(path_util.get_ndf_obj_descriptions(), 'bottle_std_centered_obj_normalized/f4851a2835228377e101b7546e3ee8a7/models/model_normalized.obj')
    # obj_model = osp.join(path_util.get_ndf_obj_descriptions(), 'bottle_centered_obj_normalized/e593aa021f3fa324530647fc03dd20dc/models/model_normalized.obj')

    scale = 1.0
    mesh = trimesh.load(obj_model, process=False)
    mesh.apply_scale(scale)

    # sample_pts = mesh1.sample(n_samples)

    # Make mesh 1 upright
    rot1 = np.eye(4)
    rot1[:3, :3] = util.make_rotation_matrix('x', np.pi / 2)
    # rot1 = np.eye(4)
    # rot1[:3, :3] = R.random().as_matrix()
    mesh.apply_transform(rot1)

    # rot2 = np.eye(4)
    # rot2[:3, :3] = R.random().as_matrix()
    # mesh.apply_transform(rot2)

    pcd = mesh.sample(2000)

    sample_pt = np.array([[-0.010, 0.020, 0.117]])

    # multiplot([pcd], osp.join(path_util.get_ndf_eval(), 'debug_viz', 'debug_grid_fig.html'))

    color = np.zeros(pcd.shape[0])

    fig = px.scatter_3d(
        x=pcd[:, 0], y=pcd[:, 1], z=pcd[:, 2], color=color
    )
    fig.update_traces(marker_color='rgba(50, 50, 50, 0.5)', selector=dict(type='scatter3d'))

    fig.add_trace(go.Scatter3d(x=sample_pt[:, 0], y=sample_pt[:, 1], z =sample_pt[:, 2], mode='markers', marker=dict(color='rgba(135, 206, 250, 0.5)')))

    # # https://stackoverflow.com/questions/62403763/how-to-add-planes-in-a-3d-scatter-plot
    # fig.add_trace(go.Surface())


    # bright_blue = [[0, '#7DF9FF'], [1, '#7DF9FF']]
    # bright_pink = [[0, '#FF007F'], [1, '#FF007F']]
    color1 = 'rgba(135, 206, 250, 0.2)'
    light_yellow = [[0, color1], [1, color1]]


    add_plane(fig, 'z', (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), 0, light_yellow)
    add_plane(fig, 'x', (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), 0, light_yellow)
    add_plane(fig, 'y', (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), 0, light_yellow)
    # # # need to add starting point of 0 to each dimension so the plane extends all the way out
    # x = np.linspace(-0.05, 0.05, 20)
    # y = np.linspace(-0.05, 0.05, 20)
    # z = 0.1 * np.ones((20, 20))

    # fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=light_yellow,  showscale=False))

    # # x = np.linspace(-0.05, 0.05, 20)
    # x = np.linspace(0.04, 0.04, 20)
    # y = np.linspace(-0.05, 0.05, 20)
    # z = np.repeat(np.linspace(-0.1, 0.1, 20).reshape(1, 20), 20, axis=0)
    # # z = 0.1 * np.ones((20, 20)) * np.linspace(-1, 1, 20)

    # print(z)

    # fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=light_yellow,  showscale=False))

    # https://plotly.com/python/3d-axes/
    # https://stackoverflow.com/questions/61693014/how-to-hide-plotly-yaxis-title-in-python

    fig.update_layout(scene = dict(
        xaxis = dict(
            gridcolor="white",
            showbackground=False,
            # showticklabels = False,
            visible = False,
            zerolinecolor="white",),
        yaxis = dict(
            gridcolor="white",
            showbackground=False,
            # showticklabels = False,
            visible = False,
            zerolinecolor="white"),
        zaxis = dict(
            # backgroundcolor="rgb(230, 230,200)",
            backgroundcolor='white',
            gridcolor="white",
            showbackground=False,
            # showticklabels = False,
            visible = False,
            zerolinecolor="white",),),
    )


    fname = osp.join(path_util.get_ndf_eval(), 'debug_viz', 'debug_grid_fig.html')

    fig.write_html(fname)
