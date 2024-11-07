from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

import os
import numpy as np

from .seasfire_unet_module import unet_features
from .components.receptive_field import compute_proto_layer_rf_info_v2

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

base_architecture_to_features = {'unet': unet_features}


#@gin.configurable(allowlist=['bottleneck_stride', 'patch_classification'])
class PPNet(nn.Module):
    def __init__(self,
                 image_size,
                 prototype_shape,
                 proto_layer_rf_info,
                 num_classes,
                 input_vars: list = None,
                 positional_vars: list = None,
                 init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck',
                 bottleneck_stride: Optional[int] = None,
                 patch_classification: bool = True,
                 norm_proto: bool =  False):

        super(PPNet, self).__init__()

        self.image_size = image_size
        self.norm_proto = norm_proto

        self.epsilon = 1e-4
        self.bottleneck_stride = bottleneck_stride
        self.patch_classification = patch_classification

        self.input_vars = list(input_vars)

        self.prototype_vectors = nn.Parameter(torch.rand(tuple(prototype_shape)), requires_grad=True)
        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    num_classes)
        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for i in range(self.num_classes):
            self.prototype_class_identity[i * num_prototypes_per_class:(i + 1) * num_prototypes_per_class, i] = 1

        assert self.num_prototypes % self.num_classes == 0

        self.num_prototypes_per_class = num_prototypes_per_class
        self.proto_layer_rf_info = proto_layer_rf_info

        self.features = base_architecture_to_features["unet"](self.input_vars, nb_classes = prototype_shape[1], pretrained=False)
        features = self.features
        
        # features_name = str(self.features).upper()
        # if features_name.startswith('VGG') or features_name.startswith('RES'):
        #     first_add_on_layer_in_channels = \
        #         [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        # elif features_name.startswith('DENSE'):
        #     first_add_on_layer_in_channels = \
        #         [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        # elif features_name.startswith('DEEPLAB'):
        #     first_add_on_layer_in_channels = \
        #         [i for i in features.modules() if isinstance(i, nn.Conv2d)][-2].out_channels
        # elif features_name.startswith('MSC'):
        #     first_add_on_layer_in_channels = \
        #         [i for i in features.base.modules() if isinstance(i, nn.Conv2d)][-2].out_channels
        # else:
        #     raise Exception(f'{features_name[:10]} base_architecture NOT implemented')
        first_add_on_layer_in_channels = \
                 [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels  
                 # put to [-2]
        add_on_layers = []

        if add_on_layers_type == 'bottleneck_pool':

            # Add conv net with stride to get the target number of patches (16x8)
            add_on_layers.append(nn.Conv2d(in_channels=first_add_on_layer_in_channels,
                                           out_channels=first_add_on_layer_in_channels,
                                           kernel_size=3, padding=1, stride=self.bottleneck_stride))
            add_on_layers.append(nn.ReLU())

        if add_on_layers_type.startswith('bottleneck'):
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    print(current_out_channels)
                    add_on_layers.append(nn.ReLU())
                else:
                    assert (current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)

        elif add_on_layers_type == 'deeplab_simple':
            self.add_on_layers = nn.Sequential(
                nn.Sigmoid()
            )
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1],
                          kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
            )

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False)  # do not use bias
        if init_weights:
            self._initialize_weights()

    @property
    def prototype_shape(self):
        return self.prototype_vectors.shape

    @property
    def num_prototypes(self):
        return self.prototype_vectors.shape[0]

    @property
    def num_classes(self):
        return self.prototype_class_identity.shape[1]

    def run_last_layer(self, prototype_activations):
        return self.last_layer(prototype_activations)

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        #print(x[0])
        #[-1]
        #print(x.shape)
        # multi-scale training (MCS)
        if isinstance(x, list):
            return [self.add_on_layers(x_scaled) for x_scaled in x]

        #print("ADD ON LAYERS")
        x = self.add_on_layers(x)
        #print(x.shape)
        return x

    #@staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)
        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)
        
        return distances

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        if self.norm_proto:
            proto = F.normalize(self.prototype_vectors, p=2, dim=1) # [nb_proton, nb_features, 1, 1]
            x = F.normalize(x, p=2, dim=1) # [nb_batches, nb_features, height, width]
        else:
            proto = self.prototype_vectors
        print(self.prototype_vectors.shape)
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = proto ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))

        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)
        xp = F.conv2d(input=x, weight=proto)
        # euclidian distance

        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        #distances = - xp
        distances = F.relu(x2_patch_sum + intermediate_result)
        return distances  # [batch , nb_proto, img_size]


    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)
        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        elif self.prototype_activation_function == 'cosin':
            return 1/(1-distances)
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x, **kwargs):
        conv_features = self.conv_features(x)
        # MCS
        if isinstance(conv_features, list):
            return [self.forward_from_conv_features(c, **kwargs) for c in conv_features]

        return self.forward_from_conv_features(conv_features, **kwargs)

    def forward_with_features(self, x, **kwargs):
        conv_features = self.conv_features(x)

        # MCS
        if isinstance(conv_features, list):
            results = [self.forward_from_conv_features(c, **kwargs) for c in conv_features]
            return [(r[0], r[1], c) for r, c in zip(results, conv_features)]

        logits, distances = self.forward_from_conv_features(conv_features, **kwargs)
        return logits, distances, conv_features

    def forward_from_conv_features(self, conv_features, return_activations=False, return_distances=False):
        if isinstance(conv_features, list):
            return [self.forward_from_conv_features(c) for c in conv_features]

        # distances.shape = (batch_size, num_prototypes, n_patches_cols, n_patches_rows)
        distances = self._l2_convolution(conv_features)
        if hasattr(self, 'patch_classification') and self.patch_classification:
            # flatten to get predictions per patch
            batch_size, num_prototypes, n_patches_cols, n_patches_rows = distances.shape

            # shape: (batch_size, n_patches_cols, n_patches_rows, num_prototypes)
            dist_view = distances.permute(0, 2, 3, 1).contiguous()
            dist_view = dist_view.reshape(-1, num_prototypes)
            prototype_activations = self.distance_2_similarity(dist_view)
            #print(prototype_activations[0,:])
            logits = self.run_last_layer(dist_view)
            #print(prototype_activations)
            #logits = torch.nn.functional.softmax(logits)
            # shape: (batch_size, n_patches_cols, n_patches_rows, num_classes)
            logits = logits.reshape(batch_size, n_patches_cols, n_patches_rows, -1)
            #print(logits)
            if return_activations:
                return logits, prototype_activations
            return logits, distances
        else:
            # original function from ProtoPNet

            # global min pooling
            min_distances = -F.max_pool2d(-distances,
                                          kernel_size=(distances.size()[2],
                                                       distances.size()[3]))
            # min_distances.shape = (batch_size, num_prototypes)
            min_distances = min_distances.view(-1, self.num_prototypes)
            prototype_activations = self.distance_2_similarity(min_distances)
            logits = self.run_last_layer(prototype_activations)
            if return_distances:
                return logits, min_distances, distances
            else:
                return logits, min_distances

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        conv_output = self.conv_features(x)

        if isinstance(conv_output, list):
            return [(c, self._l2_convolution(c)) for c in conv_output]

        distances = self._l2_convolution(conv_output)
        return conv_output, distances

    def prune_prototypes(self, prototypes_to_prune):
        '''
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        '''
        prototypes_to_keep = list(sorted(set(range(self.num_prototypes)) - set(prototypes_to_prune)))

        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...],
                                              requires_grad=True)

        # self.prototype_shape = list(self.prototype_vectors.size())
        # self.num_prototypes = self.prototype_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.ones is nn.Parameter
        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...],
                                 requires_grad=False)
        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def __repr__(self):
        # PPNet(self, features, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        incorrect_class_connection = incorrect_strength
        correct_class_connection = 1

        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

#@gin.configurable(denylist=['img_size'])
def construct_PPNet(cfg: DictConfig):
    input_vars = cfg.input_vars
    img_size=cfg.image_size
    base_architecture='unet'
    pretrained=False
    prototype_shape= cfg.prototype_shape
    num_classes=cfg.num_classes
    prototype_activation_function=cfg.prototype_activation_function
    add_on_layers_type= cfg.add_on_layers_type

    proto_layer_rf_info = []
    return PPNet(input_vars= input_vars,
                 image_size=img_size,
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type)