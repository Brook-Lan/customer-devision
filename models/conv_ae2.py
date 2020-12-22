#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2020-12-15

@author:LHQ
"""
import torch.nn as nn

from .basic_module import BasicModule


class ConvAutoEncoderV2(BasicModule):
    """ 自动编码器， 输入SIZE 为 320 * 320
    """
    def __init__(self):
        super(ConvAutoEncoderV2, self).__init__()

        # input : batch * 3 * 100 * 100
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=1, padding=1),   # batch * 8 * 50 * 50
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),   # batch * 8 * 25 * 25
                nn.BatchNorm2d(32),

                nn.Conv2d(32, 16, 3, stride=1, padding=1),   # batch * 8 * 50 * 50
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),   # batch * 8 * 25 * 25
                nn.BatchNorm2d(16),

                nn.Conv2d(16, 16, 3, stride=1, padding=1),    # batch * 8 * 25 * 25
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),   # batch * 8 * 12 * 12
                nn.BatchNorm2d(16),

                nn.Conv2d(16, 8, 3, stride=1, padding=1),    # batch * 8 * 25 * 25
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),   # batch * 8 * 12 * 12
                nn.BatchNorm2d(8),
                )

        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2),  # batch * 16 * 24 * 24  公式为:output = (input -1) * stride + outputpadding  - 2 * padding + kernelsize
                nn.ReLU(),
                nn.BatchNorm2d(16),

                nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2),  # batch * 16 * 24 * 24  公式为:output = (input -1) * stride + outputpadding  - 2 * padding + kernelsize
                nn.ReLU(),
                nn.BatchNorm2d(16),

                nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2),  # batch * 16 * 24 * 24  公式为:output = (input -1) * stride + outputpadding  - 2 * padding + kernelsize
                nn.ReLU(),
                nn.BatchNorm2d(32),

                nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2),   # batch * 3 * 100 * 100
                )
    
    def forward(self, x, output_encode=False):
        encoded = self.encoder(x)
        if output_encode:
            return encoded
        else:
            return self.decoder(encoded)


if __name__ == "__main__":
    from torchsummary import summary
    model = ConvAutoEncoderV2()
    # summary(model, (3, 400, 400))
    # summary(model, (3, 256, 256))
    summary(model, (3, 187, 187))


