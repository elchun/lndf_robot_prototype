from ndf_robot.eval.evaluate_general import EvaluateGrasp, EvaluateGraspSetup, QueryPoints
import plotly.express as px
import numpy as np
import os
import os.path as osp
import trimesh

if __name__ == '__main__':
    config_fname = 'debug_config.yml'

    setup = EvaluateGraspSetup()
    setup.load_config(config_fname)
    demo_load_dir = setup.get_demo_load_dir(obj_class='mug')

    demo_fnames = os.listdir(demo_load_dir)

    place_demo_fnames = [osp.join(demo_load_dir, fn) for fn in
        demo_fnames if 'place_demo' in fn]

        # shapenet_id
        # ee_pose_world
        # robot_joints
        # obj_pose_world
        # obj_pose_camera
        # object_pointcloud
        # depth
        # seg
        # camera_poses
        # obj_model_file
        # obj_model_file_dec
        # gripper_pts
        # rack_pointcloud_observed
        # rack_pointcloud_gt
        # rack_pointcloud_gaussian
        # rack_pointcloud_uniform
        # rack_pose_world
        # rack_contact_pose
        # shelf_pose_world
        # shelf_pointcloud_observed
        # shelf_pointcloud_uniform
        # shelf_pointcloud_gt
        # table_urdf
    place_demo_fn = place_demo_fnames[0]
    print(f'Loading demo from fname: {place_demo_fn}')
    place_data = np.load(place_demo_fn, allow_pickle=True)
    files = place_data.files
    for f in files:
        print(f)



    # rack_pcd = place_data['rack_pointcloud_uniform']
    rack_pcd = place_data['rack_pointcloud_gt']

    # cylinder_pts = QueryPoints.generate_cylinder(400, 0.02, 0.15, 'z')
    # transform = np.eye(4)
    # rot = EvaluateGrasp.make_rotation_matrix('y', 0.68)
    # trans = np.array([[0.04, 0, 0.17]]).T
    # transform[:3, :3] = rot
    # transform[:3, 3:4] = trans
    # print(transform)

    # cylinder_pcd = trimesh.PointCloud(cylinder_pts)
    # cylinder_pcd.apply_transform(transform)
    # cylinder_pts = np.asarray(cylinder_pcd.vertices)

    cylinder_pts = QueryPoints.generate_rack_arm(400)

    plot_pts = np.vstack((rack_pcd, cylinder_pts))
    color = np.concatenate([np.ones(rack_pcd.shape[0]) * 1, np.ones(cylinder_pts.shape[0]) * 2])

    fig = px.scatter_3d(
        x=plot_pts[:, 0], y=plot_pts[:, 1], z=plot_pts[:, 2], color=color)

    fig.write_html('debug.html')
    print(place_data['rack_pointcloud_uniform'].shape)
