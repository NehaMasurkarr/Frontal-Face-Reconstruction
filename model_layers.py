import torch.nn as nn
from torch.nn.init import kaiming_normal_

def initialize_weights(weight, activation_fn=None):
    if hasattr(activation_fn, "negative_slope"):
        kaiming_normal_(weight, a=activation_fn.negative_slope)
    else:
        kaiming_normal_(weight, a=0)

def create_sequential(*layers):
    sequential_model = nn.Sequential(*layers)
    for layer in reversed(layers):
        if hasattr(layer, 'out_channels'):
            sequential_model.out_channels = layer.out_channels
            break
        if hasattr(layer, 'out_features'):
            sequential_model.out_channels = layer.out_features
            break
    return sequential_model

def convolutional_block(in_channels, out_channels, kernel_size, stride=1, padding=0, activation_fn=nn.LeakyReLU(), use_batchnorm=False, init_weights=True):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
    if init_weights:
        initialize_weights(layers[-1].weight, activation_fn)
    
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    if activation_fn is not None:
        layers.append(activation_fn)  
    
    conv_block = nn.Sequential(*layers)
    conv_block.out_channels = out_channels
    return conv_block

def deconvolutional_block(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, activation_fn=nn.ReLU(), use_batchnorm=False):
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding))
    initialize_weights(layers[-1].weight, activation_fn)
    
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))    
    if activation_fn is not None:
        layers.append(activation_fn)
        
    deconv_block = nn.Sequential(*layers)
    deconv_block.out_channels = out_channels
    return deconv_block

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=1, activation_fn=nn.LeakyReLU()):
        super(ResidualBlock, self).__init__()
        
        if out_channels is None:
            out_channels = in_channels // stride
        
        self.activation_fn = activation_fn
        self.input_layer = convolutional_block(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, activation_fn=None, use_batchnorm=False)
        
        layers = []
        padding = (kernel_size - 1) // 2
        layers.append(convolutional_block(in_channels, in_channels, kernel_size, stride=1, padding=padding, activation_fn=activation_fn, use_batchnorm=True))
        layers.append(convolutional_block(in_channels, out_channels, kernel_size, stride=1, padding=padding, activation_fn=None, use_batchnorm=True))
        self.residual_layers = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x):
        output = self.activation_fn(self.residual_layers(x) + self.input_layer(x))
        return output
