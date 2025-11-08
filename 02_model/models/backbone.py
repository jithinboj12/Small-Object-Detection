import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.ops import FeaturePyramidNetwork, misc as tv_misc

def _resnet_backbone(pretrained=True, trainable_layers=3):
    # returns a resnet50 with conv1..layer4 accessible
    model = resnet50(pretrained=pretrained, norm_layer=nn.BatchNorm2d)
    # freeze layers as required
    layers_to_train = ["layer4", "layer3", "layer2"][:trainable_layers]
    for name, parameter in model.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)
    return model

class ResNetFPNBackbone(nn.Module):
    def __init__(self, pretrained=True, trainable_layers=3, out_channels=256):
        super().__init__()
        resnet = _resnet_backbone(pretrained, trainable_layers)
        # use the base layers
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # C2
        self.layer2 = resnet.layer2  # C3
        self.layer3 = resnet.layer3  # C4
        self.layer4 = resnet.layer4  # C5

        in_channels_list = [
            256, 512, 1024, 2048
        ]
        # torchvison FPN wants OrderedDict of names->channels
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels
        )

    def forward(self, x):
        # produce C2..C5 feature maps
        c1 = self.conv1(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        features = {"0": c2, "1": c3, "2": c4, "3": c5}
        # FPN yields dict of feature maps with same keys
        out = self.fpn(features)
        # ensure sorted order list
        return [out[k] for k in sorted(out.keys(), key=lambda x: int(x))]
