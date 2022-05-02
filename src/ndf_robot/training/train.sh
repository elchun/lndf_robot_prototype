# python train_vnn_occupancy_net.py --obj_class all --experiment_name  ndf_training_exp --num_epochs 100
CUDA_VISIBLE_DEVICES=7 python train_conv_occupancy_net.py --obj_class all --experiment_name  conv_occ_latent_1x100 --num_epochs 10 --iters_til_ckpt 1000 --steps_til_summary 100 --batch_size 12 

