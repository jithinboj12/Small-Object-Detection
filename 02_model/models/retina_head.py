import torch
import torch.nn as nn
import torch.nn.functional as F

class RetinaHead(nn.Module):
    def __init__(self, in_channels=256, num_anchors=9, num_classes=1, feat_channels=256, stacked_convs=4):
        super().__init__()
        # classification branch
        cls_tower = []
        bbox_tower = []
        for _ in range(stacked_convs):
            cls_tower.append(nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1, bias=True))
            cls_tower.append(nn.ReLU(inplace=True))
            bbox_tower.append(nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1, bias=True))
            bbox_tower.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_tower)
        self.bbox_tower = nn.Sequential(*bbox_tower)

        self.cls_logits = nn.Conv2d(feat_channels, num_anchors * num_classes, kernel_size=3, padding=1)
        self.bbox_pred = nn.Conv2d(feat_channels, num_anchors * 4, kernel_size=3, padding=1)

        # initialization
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # bias initialization for focal loss recommended
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, features):
        # features: list of feature maps (P3..P6 or similar)
        cls_outputs = []
        bbox_outputs = []
        for x in features:
            cls_feat = self.cls_tower(x)
            bbox_feat = self.bbox_tower(x)
            cls_out = self.cls_logits(cls_feat)
            bbox_out = self.bbox_pred(bbox_feat)
            # reshape to (N, A, C, H, W) as needed by training/inference pipeline
            cls_outputs.append(cls_out)
            bbox_outputs.append(bbox_out)
        return cls_outputs, bbox_outputs
