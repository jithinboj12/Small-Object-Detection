import torch
import torch.nn as nn
from .backbone import ResNetFPNBackbone
from .retina_head import RetinaHead
from .attention import CBAM

class RetinaNet(nn.Module):
    def __init__(self, backbone_pretrained=True, num_classes=1,
                 fpn_out_channels=256, use_cbam=False, anchor_cfg=None):
        """
        num_classes: number of object classes (not including background)
        anchor_cfg: dict for anchor generation (if needed)
        """
        super().__init__()
        self.backbone = ResNetFPNBackbone(pretrained=backbone_pretrained, out_channels=fpn_out_channels)
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam_blocks = nn.ModuleList([CBAM(fpn_out_channels) for _ in range(4)])
        else:
            self.cbam_blocks = None
        # set num_anchors (default 9 typical for 3 scales x 3 ratios)
        num_anchors = 9
        self.head = RetinaHead(in_channels=fpn_out_channels, num_anchors=num_anchors, num_classes=num_classes)

    def forward(self, x):
        # x: input images tensor (N,C,H,W)
        features = self.backbone(x)  # list of feature maps
        if self.use_cbam:
            features = [cbam(f) for cbam, f in zip(self.cbam_blocks, features)]
        cls_outs, bbox_outs = self.head(features)
        return cls_outs, bbox_outs

if __name__ == "__main__":
    # quick smoke test
    model = RetinaNet(backbone_pretrained=False, num_classes=1, use_cbam=True)
    dummy = torch.randn(2, 3, 512, 512)
    cls, bbox = model(dummy)
    print("num feature maps:", len(cls))
    for c in cls:
        print("cls shape:", c.shape)
    for b in bbox:
        print("bbox shape:", b.shape)
