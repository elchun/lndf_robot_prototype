CUDA_VISIBLE_DEVICES=0 python evaluate_ndf.py \
        --demo_exp grasp_rim_hang_handle_gaussian_precise_w_shelf \
        --object_class mug \
        --opt_iterations 500 \
        --only_test_ids \
        --rand_mesh_scale \
        --model_path multi_category_weights \
        --save_vis_per_model \
        --config eval_mug_gen \
	--exp conv_eval_latent_any_transfer_triplet_rand_margin_no_neg_margin_90000 \
        --num_iterations 100 \
        --any_pose \
#       --pybullet_viz \
#       --use_full_hand \
#		--grasp_viz \
