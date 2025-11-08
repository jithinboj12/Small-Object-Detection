import torch
import math

def generate_anchors(base_size=16, ratios=[0.5,1,2], scales=[2**0, 2**(1/3), 2**(2/3)]):
    """
    Generate anchors centered at (0,0) for given base_size, ratios and scales.
    Returns tensor (N, 4) of x1,y1,x2,y2 offsets about center.
    """
    anchors = []
    for scale in scales:
        for ratio in ratios:
            area = (base_size * scale) ** 2.0
            w = math.sqrt(area / ratio)
            h = w * ratio
            anchors.append([-w/2.0, -h/2.0, w/2.0, h/2.0])
    return torch.tensor(anchors, dtype=torch.float32)

def shift(feature_map_size, stride, anchors):
    """
    Shift anchors across the feature map (H x W) with stride.
    Returns anchors_all (A * H * W, 4)
    """
    H, W = feature_map_size
    shift_x = torch.arange(0, W * stride, step=stride)
    shift_y = torch.arange(0, H * stride, step=stride)
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
    shifts = torch.stack((shift_x.reshape(-1), shift_y.reshape(-1), shift_x.reshape(-1), shift_y.reshape(-1)), dim=1)
    A = anchors.shape[0]
    K = shifts.shape[0]
    anchors = anchors.reshape(1, A, 4) + shifts.reshape(K, 1, 4)
    anchors = anchors.reshape(K * A, 4)
    return anchors
