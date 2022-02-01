import os, os.path as osp
from ndf_robot.utils import path_util, trimesh_util, util
import numpy as np
import trimesh
import pprint

from ndf_robot.utils.eval_gen_utils import (
    process_xq_data,
)

def parse_grasp_data(grasp_data):
    pass

def make_grasp_demo_fn(shapenet_id, demo_load_dir):
    return osp.join(demo_load_dir, 'grasp_demo_%s.npz' % shapenet_id)




if __name__ == '__main__':
    obj_class = 'mug'
    demo_exp = 'grasp_rim_hang_handle_gaussian_precise_w_shelf'
    demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', obj_class, demo_exp)

    # shapenet_id = '928a383f79698c3fb6d9bc28c8d8a2c4'
    # shapenet_id = '3143a4accdc23349cac584186c95ce9b'
    # shapenet_id = '5c48d471200d2bf16e8a121e6886e18d'
    shapenet_id = '6aec84952a5ffcf33f60d03e1cb068dc'

    grasp_demo_fn = make_grasp_demo_fn(shapenet_id, demo_load_dir)

    assert osp.exists(grasp_demo_fn), 'Invalid demo fn!'

    grasp_data = np.load(grasp_demo_fn, allow_pickle=True)

    gripper_pts = grasp_data['gripper_pts_uniform']

    ##############################
    # GET FULL HAND QUERY POINTS #
    ##############################

    gripper_mesh_file = osp.join(path_util.get_ndf_descriptions(), 'franka_panda/meshes/panda_open_hand_full.obj')

    full_gripper_mesh = trimesh.load_mesh(gripper_mesh_file)
    full_gripper_pts = full_gripper_mesh.sample(500)
    # full_gripper_pts_gaussian = np.random.normal(size=(500,3))
    # full_gripper_pts_pcd = trimesh.PointCloud(gripper_pts)
    # full_gripper_pts_bb = full_gripper_mesh.bounding_box_oriented
    # full_gripper_pts_uniform = full_gripper_pts_bb.sample_volume(500)
    full_gripper_pts_uniform = trimesh.sample.volume_mesh(full_gripper_mesh, 1000)
    full_gripper_pts_pcd = trimesh.PointCloud(full_gripper_pts_uniform)

    gripper_pts = full_gripper_pts_uniform

    # Load object points 
    data = grasp_data
    demo_obj_pts = data['object_pointcloud']  # observed shape point cloud at start
    demo_pts_mean = np.mean(demo_obj_pts, axis=0)
    inliers = np.where(np.linalg.norm(demo_obj_pts - demo_pts_mean, 2, 1) < 0.2)[0]
    demo_obj_pts = demo_obj_pts[inliers]
    demo_obj_pcd = trimesh.PointCloud(demo_obj_pts)

    rix = np.random.permutation(demo_obj_pcd.shape[0])
    demo_obj_pcd_disp = demo_obj_pcd[rix[:int(demo_obj_pcd.shape[0]/5)]]
    demo_obj_pts = np.asarray(demo_obj_pcd.vertices)


    pick_demo_obj_pose = data['obj_pose_world']
    demo_obj_pcd = trimesh.PointCloud(demo_obj_pts)
    demo_obj_mat = util.matrix_from_pose(util.list2pose_stamped(pick_demo_obj_pose))
    # demo_obj_pcd.apply_transform(demo_obj_mat)
    # demo_obj_pts = np.asarray(demo_obj_pcd.vertices)



    # Move gripper to pose
    demo_gripper_pts = gripper_pts
    # demo_gripper_pts = data['gripper_pts_uniform']
    demo_gripper_pcd = trimesh.PointCloud(demo_gripper_pts)
    demo_ee_mat = util.matrix_from_pose(util.list2pose_stamped(data['ee_pose_world']))
    demo_gripper_pcd.apply_transform(demo_ee_mat)

    # New
    demo_gripper_pcd.apply_translation([0, 0, 0.11])

    demo_gripper_pts = np.asarray(demo_gripper_pcd.vertices)

    # print(pick_demo_obj_pose)

    # trimesh_util.trimesh_show([demo_gripper_pts, demo_obj_pts]) 

    scene = trimesh_util.trimesh_show([demo_gripper_pts, demo_obj_pts], show=False)
    grasp_sph = trimesh.creation.uv_sphere(0.005)
    grasp_sph.apply_transform(demo_ee_mat)

    obj_sph = trimesh.creation.uv_sphere(0.005)
    obj_sph.apply_transform(demo_obj_mat)

    scene.add_geometry([grasp_sph, obj_sph])
    scene.show()

    # demo_object_path = osp.join(path_util.get_ndf_descriptions(), 'demo_objects/mug_centered_obj_normalized/%s/models/model_normalized.obj' % shapenet_id)
    # # demo_object_path = grasp_data['obj_model_file']
    # print(demo_object_path)
    # demo_object_mesh = trimesh.load_mesh(demo_object_path)
    # demo_object_pcd = demo_object_mesh.sample(5000)

    # trimesh_util.trimesh_show([demo_object_pcd, full_gripper_pts_uniform])




    #TODO Get object, gripper pts, pose --> apply pose to get points in appropriate location

    #TODO Sample from points









#     print("HERE")

#     # get filenames of all the demo files
#     demo_filenames = os.listdir(demo_load_dir)
#     assert len(demo_filenames), 'No demonstrations found in path: %s!' % demo_load_dir 

#     # strip the filenames to properly pair up each demo file
#     # use the grasp names as a reference
#     grasp_demo_filenames_orig = [osp.join(demo_load_dir, fn) for fn in demo_filenames if 'grasp_demo' in fn]

#     place_demo_filenames = []
#     grasp_demo_filenames = []
#     for i, fname in enumerate(grasp_demo_filenames_orig):
#         shapenet_id_npz = fname.split('/')[-1].split('grasp_demo_')[-1]
#         place_fname = osp.join('/'.join(fname.split('/')[:-1]), 'place_demo_' + shapenet_id_npz)
#         if osp.exists(place_fname):
#             grasp_demo_filenames.append(fname)
#             place_demo_filenames.append(place_fname)
#         else:
#             print('Could not find corresponding placement demo: %s, skipping ' % place_fname)
    

#     for i, fname in enumerate(grasp_demo_filenames):
#         print('Loading demo from fname: %s' % fname)
#         grasp_demo_fn = grasp_demo_filenames[i]
#         grasp_data = np.load(grasp_demo_fn, allow_pickle=True)
    
#     print(make_grasp_demo_fn(shapenet_id, demo_load_dir))    

#     # optimizer_gripper_pts, rack_optimizer_gripper_pts, shelf_optimizer_gripper_pts = process_xq_data(grasp_data, place_data, shelf=load_shelf)
    
#     # pprint.pprint(place_demo_filenames)
#     # pprint.pprint(grasp_demo_filenames)

#     # pprint.pprint(demo_filenames)

# # TODO: make parser for grasp data to get shape and pose, then substitute new point cloud