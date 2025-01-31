import torch
import torch.nn as nn
from blocks import convolutional_block, deconvolutional_block, ResidualBlock, create_sequential

class LocalFeatureExtractor(nn.Module):
    def __init__(self):
        super(LocalFeatureExtractor, self).__init__()
        encoder_channels = [64, 128, 256, 512]
        decoder_channels = [256, 128, 64]

        # Encoder blocks
        self.encoder_block_0 = create_sequential(
            convolutional_block(in_channels=3, out_channels=encoder_channels[0], kernel_size=3, stride=1, padding=1,
                                activation_fn=nn.LeakyReLU(), use_batchnorm=True),
            ResidualBlock(encoder_channels[0], activation_fn=nn.LeakyReLU())
        )

        self.encoder_block_1 = create_sequential(
            convolutional_block(in_channels=encoder_channels[0], out_channels=encoder_channels[1], kernel_size=3, stride=2, padding=1,
                                activation_fn=nn.LeakyReLU(), use_batchnorm=True),
            ResidualBlock(encoder_channels[1], activation_fn=nn.LeakyReLU())
        )

        self.encoder_block_2 = create_sequential(
            convolutional_block(in_channels=encoder_channels[1], out_channels=encoder_channels[2], kernel_size=3, stride=2, padding=1,
                                activation_fn=nn.LeakyReLU(), use_batchnorm=True),
            ResidualBlock(encoder_channels[2], activation_fn=nn.LeakyReLU())
        )

        self.encoder_block_3 = create_sequential(
            convolutional_block(in_channels=encoder_channels[2], out_channels=encoder_channels[3], kernel_size=3, stride=2, padding=1,
                                activation_fn=nn.LeakyReLU(), use_batchnorm=True),
            *[ResidualBlock(encoder_channels[3], activation_fn=nn.LeakyReLU()) for _ in range(2)]
        )

        # Decoder blocks
        self.decoder_block_0 = deconvolutional_block(in_channels=encoder_channels[3], out_channels=decoder_channels[0],
                                                     kernel_size=3, stride=2, padding=1, output_padding=1,
                                                     activation_fn=nn.ReLU(), use_batchnorm=True)

        self.decoder_process_0 = create_sequential(
            convolutional_block(in_channels=decoder_channels[0] + self.encoder_block_2.out_channels, out_channels=decoder_channels[0],
                                kernel_size=3, stride=1, padding=1,
                                activation_fn=nn.LeakyReLU(), use_batchnorm=True),
            ResidualBlock(decoder_channels[0], activation_fn=nn.LeakyReLU())
        )

        self.decoder_block_1 = deconvolutional_block(in_channels=decoder_channels[0], out_channels=decoder_channels[1],
                                                     kernel_size=3, stride=2, padding=1, output_padding=1,
                                                     activation_fn=nn.ReLU(), use_batchnorm=True)

        self.decoder_process_1 = create_sequential(
            convolutional_block(in_channels=decoder_channels[1] + self.encoder_block_1.out_channels, out_channels=decoder_channels[1],
                                kernel_size=3, stride=1, padding=1,
                                activation_fn=nn.LeakyReLU(), use_batchnorm=True),
            ResidualBlock(decoder_channels[1], activation_fn=nn.LeakyReLU())
        )

        self.decoder_block_2 = deconvolutional_block(in_channels=decoder_channels[1], out_channels=decoder_channels[2],
                                                     kernel_size=3, stride=2, padding=1, output_padding=1,
                                                     activation_fn=nn.ReLU(), use_batchnorm=True)

        self.decoder_process_2 = create_sequential(
            convolutional_block(in_channels=decoder_channels[2] + self.encoder_block_0.out_channels, out_channels=decoder_channels[2],
                                kernel_size=3, stride=1, padding=1,
                                activation_fn=nn.LeakyReLU(), use_batchnorm=True),
            ResidualBlock(decoder_channels[2], activation_fn=nn.LeakyReLU())
        )

        self.final_output = convolutional_block(in_channels=decoder_channels[2], out_channels=3,
                                                kernel_size=1, stride=1, padding=0,
                                                activation_fn=nn.Tanh(), use_batchnorm=False)

    def forward(self, x):
        # Encoding
        encoded_0 = self.encoder_block_0(x)
        encoded_1 = self.encoder_block_1(encoded_0)
        encoded_2 = self.encoder_block_2(encoded_1)
        encoded_3 = self.encoder_block_3(encoded_2)

        # Decoding
        decoded_0 = self.decoder_block_0(encoded_3)
        processed_0 = self.decoder_process_0(torch.cat([decoded_0, encoded_2], dim=1))
        decoded_1 = self.decoder_block_1(processed_0)
        processed_1 = self.decoder_process_1(torch.cat([decoded_1, encoded_1], dim=1))
        decoded_2 = self.decoder_block_2(processed_1)
        processed_2 = self.decoder_process_2(torch.cat([decoded_2, encoded_0], dim=1))

        final_output_image = self.final_output(processed_2)

        assert final_output_image.shape == x.shape, f"Output shape {final_output_image.shape} does not match input shape {x.shape}"
        return final_output_image, processed_2
