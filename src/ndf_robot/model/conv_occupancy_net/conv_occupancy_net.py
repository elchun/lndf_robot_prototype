"""
Based on implementation from: https://pengsongyou.github.io/conv_onet
"""
import torch
import torch.nn as nn

from ndf_robot.model.conv_occupancy_net.encoder import (pointnet, pointnetpp)
from ndf_robot.model.conv_occupancy_net import decoder


# Encoder dictionary
encoder_dict = {
    'pointnet_local_pool': pointnet.LocalPoolPointnet,
    'pointnet_crop_local_pool': pointnet.PatchLocalPoolPointnet,
    'pointnet_plus_plus': pointnetpp.PointNetPlusPlus,
    # 'voxel_simple_local': voxels.LocalVoxelEncoder,
}

# Decoder dictionary
decoder_dict = {
    'simple_local': decoder.LocalDecoder,
    'simple_local_crop': decoder.PatchLocalDecoder,
    'simple_local_point': decoder.LocalPointDecoder
}

class ConvolutionalOccupancyNetwork(nn.Module):
    """
    Convolutional Occupancy Network Class

    Args:
        latent_dim (int): dim of resnet block?
        encoder_type (str): type of encoder network to use
        sigmoid (boolean): use sigmoid in decoder
        return_features (boolean): True to return features in decoder
        acts (str): What type of activations to return select from ('all', 'inp', 'first_rn', 'inp_first_rn')
        scaling (float): How much to scale input by
    """
    def __init__(self,
                 latent_dim,
                 encoder_type='pointnet_local_pool',
                 decoder_type='simple_local',
                 sigmoid=True,
                 return_features=False, 
                 acts='all',
                 scaling=10.0):
        super().__init__()

        self.latent_dim = latent_dim
        self.scaling = scaling  # scaling up the point cloud/query points to be larger helps
        self.return_features = return_features

        # Options are:
        #   'pointnet_local_pool'
        #   'pointnet_crop_local_pool'
        #   'pointnet_plus_plus'
        if encoder_type in encoder_dict:
            self.encoder = encoder_dict[encoder_type](c_dim=latent_dim)
        else: raise ValueError("Invalid Decoder")
        
        if decoder_type in decoder_dict:
            # TODO: Add arguments to decoder
            self.decoder = decoder_dict[decoder_type](dim=3, z_dim=latent_dim, c_dim=0) 
        else: raise ValueError("Invalid Decoder")

        # if model_type == 'dgcnn':
        #     self.model_type = 'dgcnn'
        #     self.encoder = VNN_DGCNN(c_dim=latent_dim) # modified resnet-18
        # else:
        #     self.model_type = 'pointnet'
        #     self.encoder = VNN_ResnetPointnet(c_dim=latent_dim) # modified resnet-18

        # self.decoder = DecoderInner(dim=3, z_dim=latent_dim, c_dim=0, hidden_size=latent_dim, leaky=True, sigmoid=sigmoid, return_features=return_features, acts=acts)


    def forward(self, input):
        """
        Forward pass through network

        Args:
            input (dict): Dictionary with keys
                'point_cloud' --> maps to points to encode
                'coords' --> query point coordinates

        Returns:
            dict: Dictionary with keys
                'occ' --> Occupancy prediction of query points
                'features' --> activations of query points???
        """
        out_dict = {}

        enc_in = input['point_cloud'] * self.scaling 
        query_points = input['coords'] * self.scaling 

        z = self.encoder(enc_in)

        if self.return_features:
            out_dict['occ'], out_dict['features'] = self.decoder(query_points, z)
            # out_dict['occ'] = self.decoder(query_points, z)
            # out_dict['features'] = None
        else:
            out_dict['occ'] = self.decoder(query_points, z)

        return out_dict

    def forward(self, p, inputs, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        p_r = self.decode(p, c, **kwargs)
        return p_r

    def extract_latent(self, input):
        """
        Get latent features from encoder based on input pointcloud

        Args:
            input (dict): Contains key 'point_cloud' describing 
                pointcloud to get activations for

        Returns:
            Concatenated activations from network _description_
        """
        enc_in = input['point_cloud'] * self.scaling 
        z = self.encoder(enc_in)
        return z

    def forward_latent(self, z, coords):
        """
        Get decoder activations from decoder network

        Args:
            z (???): Encoder activations
            coords (???): Query points

        Returns:
            ???: Concatenated activations of decoder
        """
        out_dict = {}
        coords = coords * self.scaling 
        out_dict['occ'], out_dict['features'] = self.decoder(coords, z)

        return out_dict['features']