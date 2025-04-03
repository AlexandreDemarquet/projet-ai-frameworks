import torchvision.models as models
import torch

mobilenet = models.mobilenet_v3_small()

model = torch.nn.Sequential(
    mobilenet.features,
    torch.nn.AdaptiveAvgPool2d(output_size=1),
    torch.nn.Flatten()
    )


