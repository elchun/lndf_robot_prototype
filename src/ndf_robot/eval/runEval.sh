CUDA_VISIBLE_DEVICES=0 python evaluate_ndf_v2.py \
        --demo_exp grasp_rim_hang_handle_gaussian_precise_w_shelf \
        --object_class mug \
        --opt_iterations 500 \
        --only_test_ids \
        --rand_mesh_scale \
        --model_path multi_category_weights \
        --save_vis_per_model \
        --config eval_mug_gen \
        --exp test_mug_eval \
 		--num_iterations 100 
#		--use_gripper_occ 
#       --pybullet_viz \
#       --use_full_hand \
#		--grasp_viz \
