# python train_vnn_occupancy_net.py --obj_class all --experiment_name  ndf_training_exp --num_epochs 100
python train_conv_occupancy_net.py --obj_class all --experiment_name  conv_occ_exp --num_epochs 100 --iters_til_ckpt 100 --steps_til_summary 100

