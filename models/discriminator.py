import torch
from torch import nn


class DCGanDiscriminator(nn.Module):
    def __init__(self):
        super(DCGanDiscriminator, self).__init__()
        # Model hyperparameters
        num_filters = [512, 256, 128, 64]
        strides = [2, 2, 2, 2]
        input_vector_channel = [3]
        leaky_relu_slope = 0.2

        layer_list = []
        self.inp_dim = input_vector_channel[0]
        all_units = num_filters + input_vector_channel
        kernel_size = 4

        layer_list.extend([nn.Conv2d(all_units[-1], all_units[-2], kernel_size, strides[-1], 1, bias = False),
                        nn.LeakyReLU(leaky_relu_slope, inplace=True)])
        for i in range(len(all_units) - 2, 0, -1):
            temp = [nn.Conv2d(all_units[i], all_units[i-1], kernel_size, strides[i-1], padding=1, bias=False),
                    nn.BatchNorm2d(all_units[i-1]),
                    nn.LeakyReLU(leaky_relu_slope, inplace = True)]
            layer_list.extend(temp)
        layer_list.extend([nn.Conv2d(all_units[i-1], 1, kernel_size, 1, 0, bias = False)])
        
        self.discriminator = nn.Sequential(*layer_list)
        # Initialize weights
        self.discriminator.apply(self.init_weights)

    def init_weights(self, layer):
        # Weight initialization paramters taken from DC GAN original paper by Alec Radford
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
        if isinstance(layer, nn.BatchNorm2d):
            nn.init.normal_(layer.weight.data, 1.0, 0.02)
            nn.init.constant_(layer.bias.data, 0)


    def forward(self, x):
        out = self.discriminator(x)
        return out