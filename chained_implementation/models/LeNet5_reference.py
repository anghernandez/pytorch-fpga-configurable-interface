import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.Tanh(),

            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0
            ),
            nn.Tanh(),

            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                in_features=16 * 5 * 5,
                out_features=120
            ),
            nn.Tanh(),

            nn.Linear(
                in_features=120,
                out_features=84
            ),
            nn.Tanh(),

            nn.Linear(
                in_features=84,
                out_features=num_classes
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

