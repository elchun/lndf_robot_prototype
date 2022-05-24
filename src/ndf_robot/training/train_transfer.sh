CUDA_VISIBLE_DEVICES=4 python train_conv_occupancy_net.py --obj_class all --experiment_name  conv_occ_latent_transfer_rand_coords_margin_no_neg_margin --num_epochs 12 --iters_til_ckpt 1000 --steps_til_summary 100 --batch_size 10 --checkpoint_path ndf_vnn/conv_occ_exp_archive/checkpoints/model_epoch_0099_iter_747100.pth --triplet_loss

