import torch
from torch import nn

class DCGanGenerator(nn.Module):
    def __init__(self):
        super(DCGanGenerator, self).__init__()
        # Model hyperparameters
        num_filters = [512, 256, 128, 64]
        latent_dim = 100
        strides = [2, 2, 2, 2]
        input_vector_channel = [3]

        layer_list = []
        self.inp_dim = input_vector_channel[0]
        all_units = num_filters + input_vector_channel
        kernel_size = 4

        layer_list.extend([nn.ConvTranspose2d(latent_dim, all_units[0], kernel_size, 1, 0, bias = False),
                        nn.BatchNorm2d(all_units[0]),
                        nn.ReLU(inplace=True)])
        for i in range(len(all_units) - 2):
            temp = [nn.ConvTranspose2d(all_units[i], all_units[i+1], kernel_size, strides[i], padding=1, bias=False),
                    nn.BatchNorm2d(all_units[i+1]),
                    nn.ReLU(inplace = True)]
            layer_list.extend(temp)
        layer_list.extend([nn.ConvTranspose2d(all_units[i+1], all_units[i+2], kernel_size, strides[i+1], padding=1, bias = False),
                         nn.Tanh()])
        
        self.generator = nn.Sequential(*layer_list)
        # Initialize weights
        self.generator.apply(self.init_weights)

    def init_weights(self, layer):
        # Weight initialization paramters taken from DC GAN original paper by Alec Radford
        if isinstance(layer, nn.ConvTranspose2d):
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
        if isinstance(layer, nn.BatchNorm2d):
            nn.init.normal_(layer.weight.data, 1.0, 0.02)
            nn.init.constant_(layer.bias.data, 0)


    def forward(self, x):
        x = x.view(x.shape[0], x.shape[-1],1,1)
        out = self.generator(x)
        return out