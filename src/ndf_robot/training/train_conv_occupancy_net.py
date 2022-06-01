import sys
import os, os.path as osp
import configargparse
import torch
from torch.utils.data import DataLoader
from torch import nn

import ndf_robot.model.vnn_occupancy_net.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
import ndf_robot.model.conv_occupancy_net.conv_occupancy_net as conv_occupancy_network

# from ndf_robot.training import summaries, losses, training, dataio, config
from ndf_robot.training import summaries, losses, training
from ndf_robot.training import dataio_conv as dataio
# from ndf_robot.training import dataio as dataio

from ndf_robot.utils import path_util
from ndf_robot.training.util import make_unique_path_to_dir


if __name__ == '__main__':
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--logging_root', type=str, default=osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn'), help='root for logging')
    p.add_argument('--obj_class', type=str, required=True,
                help='bottle, mug, bowl, all')
    p.add_argument('--experiment_name', type=str, required=True,
                help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

    p.add_argument('--sidelength', type=int, default=128)

    # General training options
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
    # p.add_argument('--lr', type=float, default=1e-3, help='learning rate. default=5e-5')
    p.add_argument('--num_epochs', type=int, default=100,
                help='Number of epochs to train for.')
    # p.add_argument('--num_epochs', type=int, default=40001,
    #                help='Number of epochs to train for.')

    p.add_argument('--epochs_til_ckpt', type=int, default=5,
                help='Time interval in seconds until checkpoint is saved.')
    # p.add_argument('--epochs_til_ckpt', type=int, default=10,
    #                help='Time interval in seconds until checkpoint is saved.')
    p.add_argument('--steps_til_summary', type=int, default=500,
                help='Time interval in seconds until tensorboard summary is saved.')
    p.add_argument('--iters_til_ckpt', type=int, default=10000,
                help='Training steps until save checkpoint')

    p.add_argument('--depth_aug', action='store_true', help='depth_augmentation')
    p.add_argument('--multiview_aug', action='store_true', help='multiview_augmentation')

    p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
    p.add_argument('--dgcnn', action='store_true', help='If you want to use a DGCNN encoder instead of pointnet (requires more GPU memory)')
    # p.add_argument('--conv', action='store_true', help='If you want to train convolutional occ instead of non-convolutional')

    p.add_argument('--triplet_loss', action='store_true', help='Run triplet loss on'
        + ' activations')

    opt = p.parse_args()

    train_dataset = dataio.JointOccTrainDataset(128, depth_aug=opt.depth_aug, multiview_aug=opt.multiview_aug, obj_class=opt.obj_class)
    val_dataset = dataio.JointOccTrainDataset(128, phase='val', depth_aug=opt.depth_aug, multiview_aug=opt.multiview_aug, obj_class=opt.obj_class)


    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                drop_last=True, num_workers=6)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True,
                                drop_last=True, num_workers=4)

    # model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=64).cuda()
    model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=32, return_features=True).cuda()
    # model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=32, return_features=True, acts='last').cuda()
    # model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=32, return_features=True, acts='first_net').cuda()
    # model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=16, return_features=True).cuda()

    print(model)

    if opt.checkpoint_path is not None:
        checkpoint_path = osp.join(path_util.get_ndf_model_weights(), opt.checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path))

    # Can use if have multiple gpus (best to not use for now cuz it increases complexity)
    # model_parallel = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5])
    model_parallel = model

    # Define the loss
    summary_fn = summaries.occupancy_net
    root_path = os.path.join(opt.logging_root, opt.experiment_name)

    root_path = make_unique_path_to_dir(root_path)


    ### Run train function ###
    if opt.triplet_loss:
        # loss_fn = val_loss_fn = losses.custom_rotated_triplet
        # loss_fn = val_loss_fn = losses.rotated_log
        loss_fn = val_loss_fn = losses.rotated_triplet_log
        training.train_conv_triplet(model=model_parallel, train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader, epochs=opt.num_epochs, lr=opt.lr, 
            steps_til_summary=opt.steps_til_summary, 
            epochs_til_checkpoint=opt.epochs_til_ckpt,
            model_dir=root_path, loss_fn=loss_fn, iters_til_checkpoint=opt.iters_til_ckpt, 
            summary_fn=summary_fn,clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True)
    else:
        loss_fn = val_loss_fn = losses.rotated_margin
        training.train_conv(model=model_parallel, train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader, epochs=opt.num_epochs, lr=opt.lr, 
            steps_til_summary=opt.steps_til_summary, 
            epochs_til_checkpoint=opt.epochs_til_ckpt,
            model_dir=root_path, loss_fn=loss_fn, iters_til_checkpoint=opt.iters_til_ckpt, 
            summary_fn=summary_fn,clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True)

    # Default training for reference
    # training.train(model=model_parallel, train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=opt.num_epochs,
    #                lr=opt.lr, steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
    #                model_dir=root_path, loss_fn=loss_fn, iters_til_checkpoint=opt.iters_til_ckpt, summary_fn=summary_fn,
    #                clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True)
