import torch
from torch import nn

# Description of convolution neural network architecture
ARCHITECTURE_DESCRIPTION = [
    (7, 64, 2, 3),
    "MaxPool",
    (3, 192, 1, 1),
    "MaxPool",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "MaxPool",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "MaxPool",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class ConvBlock(nn.Module):
    """
    Convolutional block component: Convolutional -> batch normalisation -> leaky relu
    ...
    Attributes
    ----------
    conv: nn.Module
        Convolutional 2-dimensional layer
    batch_norm: nn.Module
        2-dimensional batch normalisation
    leaky_relu: nn.Module
        Leaky relu with negative slope = 0.1
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              bias=False,
                              **kwargs)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor):
        return self.leaky_relu(self.batch_norm(self.conv(x)))


class Yolo(nn.Module):
    """
    Yolo version 1 model architecture:
     darknet(deep convolutional neural network) -> flattener -> fully connected layers for output
    ...
    Attributes
    ----------
    in_channels: int
        Input number of channels
    darknet: nn.Module
        Deep convolutional neural network for creating feature maps
    flattener: nn.Module
        Flattens result of darknet
    fully_connected_net: nn.Module
        Fully connected layers for creating yolo output (classes, object probability, x, y, width, height)
    """
    def __init__(self,
                 in_channels: int = 3,
                 **kwargs):
        super(Yolo, self).__init__()
        self.in_channels = in_channels
        self.darknet = self.__create_darknet()
        self.flattener = nn.Flatten(start_dim=1, end_dim=-1)
        self.fully_connected_net = self.__create_fcs(**kwargs)

    def forward(self,
                x: torch.Tensor):
        output = self.darknet(x)
        output = self.flattener(output)
        output = self.fully_connected_net(output)
        return output

    def __create_darknet(self):
        layers = []
        in_channels = self.in_channels
        for x in ARCHITECTURE_DESCRIPTION:
            if isinstance(x, tuple):
                layers.append(
                    ConvBlock(
                        in_channels=in_channels,
                        out_channels=x[1],
                        kernel_size=x[0],
                        stride=x[2],
                        padding=x[3]
                    )
                )
                in_channels = x[1]
            elif x == "MaxPool":
                layers.append(
                    nn.MaxPool2d(
                        kernel_size=2,
                        stride=2
                    )
                )
            elif isinstance(x, list):
                conv1 = x[0]
                conv2 = x[1]
                n_repeats = x[2]

                for _ in range(n_repeats):
                    layers.append(
                        ConvBlock(
                            in_channels=in_channels,
                            out_channels=conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3]
                        )
                    )
                    layers.append(
                        ConvBlock(
                            in_channels=conv1[1],
                            out_channels=conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3]
                        )
                    )
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    @staticmethod
    def __create_fcs(split_size, n_bboxes, n_classes):
        s, b, c = split_size, n_bboxes, n_classes
        return nn.Sequential(
            nn.Linear(1024 * s * s, 4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, s * s * (c + b * 5))
        )


if __name__ == "__main__":
    tens = torch.randn(10, 3, 448, 448)
    yolo = Yolo(split_size=7, n_bboxes=2, n_classes=20)
    assert list(yolo(tens).shape) == [10, 7 * 7 * 30]
    tens2 = torch.randn(10, 3, 448, 448)
    yolo2 = Yolo(split_size=7, n_bboxes=2, n_classes=1)
    assert list(yolo2(tens2).shape) == [10, 7 * 7 * 11]
