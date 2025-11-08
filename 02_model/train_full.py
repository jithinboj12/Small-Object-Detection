import os
import sys
import yaml
import time
import json
import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# import your model module from 02_model/models
# Ensure that 02_model is in PYTHONPATH or call this script from repo root.
try:
    from models.retina_net import RetinaNet
except Exception as e:
    print("Could not import models.retina_net. Make sure 02_model/models exists and PYTHONPATH includes repo root.")
    raise

# -------------------------
# Utilities
# -------------------------
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def xyxy_to_xywh(boxes):
    # boxes: (N,4) x1,y1,x2,y2 -> cx,cy,w,h
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1)
    h = (y2 - y1)
    return torch.stack([cx, cy, w, h], dim=-1)

def iou_xyxy(boxes1, boxes2):
    # boxes1: (N,4), boxes2: (M,4). returns (N,M)
    # boxes in x1,y1,x2,y2
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N,M,2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N,M,2)
    wh = (rb - lt).clamp(min=0)  # (N,M,2)
    inter = wh[:,:,0] * wh[:,:,1]
    area1 = (boxes1[:,2]-boxes1[:,0]) * (boxes1[:,3]-boxes1[:,1])
    area2 = (boxes2[:,2]-boxes2[:,0]) * (boxes2[:,3]-boxes2[:,1])
    union = area1[:,None] + area2[None,:] - inter
    return inter / union

def encode_boxes(anchors, gt_boxes):
    # Encode gt relative to anchors using (tx,ty,tw,th)
    # anchors & gt_boxes in x1,y1,x2,y2
    # returns (N,4)
    anchors = anchors.float()
    a_cx = (anchors[:,0] + anchors[:,2]) / 2.0
    a_cy = (anchors[:,1] + anchors[:,3]) / 2.0
    a_w = anchors[:,2] - anchors[:,0]
    a_h = anchors[:,3] - anchors[:,1]

    g_cx = (gt_boxes[:,0] + gt_boxes[:,2]) / 2.0
    g_cy = (gt_boxes[:,1] + gt_boxes[:,3]) / 2.0
    g_w = gt_boxes[:,2] - gt_boxes[:,0]
    g_h = gt_boxes[:,3] - gt_boxes[:,1]

    tx = (g_cx - a_cx) / (a_w + 1e-6)
    ty = (g_cy - a_cy) / (a_h + 1e-6)
    tw = torch.log((g_w + 1e-6) / (a_w + 1e-6))
    th = torch.log((g_h + 1e-6) / (a_h + 1e-6))
    return torch.stack([tx, ty, tw, th], dim=1)

def decode_boxes(anchors, deltas):
    # inverse of encode_boxes
    a_cx = (anchors[:,0] + anchors[:,2]) / 2.0
    a_cy = (anchors[:,1] + anchors[:,3]) / 2.0
    a_w = anchors[:,2] - anchors[:,0]
    a_h = anchors[:,3] - anchors[:,1]
    dx = deltas[:,0]
    dy = deltas[:,1]
    dw = deltas[:,2]
    dh = deltas[:,3]
    gx = dx * a_w + a_cx
    gy = dy * a_h + a_cy
    gw = torch.exp(dw) * a_w
    gh = torch.exp(dh) * a_h
    x1 = gx - gw / 2.0
    y1 = gy - gh / 2.0
    x2 = gx + gw / 2.0
    y2 = gy + gh / 2.0
    return torch.stack([x1,y1,x2,y2], dim=1)

# -------------------------
# Anchor generation
# -------------------------
def generate_base_anchors(base_size=16, ratios=None, scales=None):
    if ratios is None:
        ratios = [0.5, 1.0, 2.0]
    if scales is None:
        # default small-object focused scales
        scales = [1.0, 1.2599, 1.5874]  # 2**(0), 2**(1/3), 2**(2/3)
    anchors = []
    for scale in scales:
        for ratio in ratios:
            area = (base_size * scale) ** 2.0
            w = math.sqrt(area / ratio)
            h = w * ratio
            anchors.append([-w/2.0, -h/2.0, w/2.0, h/2.0])
    return torch.tensor(anchors, dtype=torch.float32)  # (A,4)

def shift_anchors(feat_h, feat_w, stride, base_anchors, device):
    # base_anchors: (A,4) centered at 0
    shift_x = torch.arange(0, feat_w * stride, step=stride, device=device)
    shift_y = torch.arange(0, feat_h * stride, step=stride, device=device)
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
    shifts = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1),
                          shift_x.reshape(-1), shift_y.reshape(-1)], dim=1)  # (K,4)
    A = base_anchors.shape[0]
    K = shifts.shape[0]
    anchors = base_anchors.reshape(1, A, 4) + shifts.reshape(K, 1, 4)
    anchors = anchors.reshape(K * A, 4)
    return anchors  # (K*A,4)

def generate_pyramid_anchors(image_size, pyramid_levels, strides, base_size, ratios, scales, device):
    """
    Create anchors for each FPN level given expected feature map sizes computed from image_size.
    pyramid_levels: list like [3,4,5,6] corresponding to P3..P6
    strides: list of ints corresponding to each level, e.g., [8,16,32,64]
    Returns:
      anchors_all: concatenated anchors (N,4) on device
      anchors_per_level: list of tensors per level
    """
    anchors_per_level = []
    base_anchors = generate_base_anchors(base_size=base_size, ratios=ratios, scales=scales).to(device)
    for level, stride in zip(pyramid_levels, strides):
        feat_h = math.ceil(image_size / stride)
        feat_w = math.ceil(image_size / stride)
        anchors = shift_anchors(feat_h, feat_w, stride, base_anchors, device)
        anchors_per_level.append(anchors)
    anchors_all = torch.cat(anchors_per_level, dim=0)
    return anchors_all, anchors_per_level

# -------------------------
# Dataset (CSV or COCO)
# -------------------------
class SmallObjectDataset(Dataset):
    """
    Supports two formats:
    - CSV: rows with image_path,x1,y1,x2,y2,class_id
      (one bounding box per row). If multiple boxes per image, repeat image_path rows.
    - COCO JSON: standard COCO annotation file + images under dataset root.
    """
    def __init__(self, annotations_path, images_root=None, img_size=512, transform=None):
        self.img_size = img_size
        self.transform = transform
        self.samples = []  # list of dicts: {'image_path':..., 'boxes':Tensor(N,4), 'labels':Tensor(N)}
        self.images_root = images_root or ""
        if annotations_path.lower().endswith('.json'):
            self._load_coco(annotations_path)
        else:
            self._load_csv(annotations_path)

    def _load_csv(self, path):
        import csv
        per_image = {}
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                img_path = row[0]
                x1, y1, x2, y2, cls = map(float, row[1:6])
                if img_path not in per_image:
                    per_image[img_path] = {'boxes': [], 'labels': []}
                per_image[img_path]['boxes'].append([x1, y1, x2, y2])
                per_image[img_path]['labels'].append(int(cls))
        for img_path, v in per_image.items():
            boxes = torch.tensor(v['boxes'], dtype=torch.float32)
            labels = torch.tensor(v['labels'], dtype=torch.long)
            self.samples.append({'image_path': img_path, 'boxes': boxes, 'labels': labels})

    def _load_coco(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        id2file = {img['id']: img['file_name'] for img in data['images']}
        anns_per_image = defaultdict(list)
        for ann in data['annotations']:
            img_id = ann['image_id']
            bbox = ann['bbox']  # COCO: x,y,w,h
            x1 = bbox[0]; y1 = bbox[1]; x2 = x1 + bbox[2]; y2 = y1 + bbox[3]
            cls = ann['category_id']
            anns_per_image[img_id].append([x1, y1, x2, y2, cls])
        for img in data['images']:
            fid = img['id']
            file_name = img['file_name']
            anns = anns_per_image.get(fid, [])
            if len(anns)==0:
                boxes = torch.zeros((0,4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.long)
            else:
                boxes = torch.tensor([[a[0],a[1],a[2],a[3]] for a in anns], dtype=torch.float32)
                labels = torch.tensor([int(a[4]) for a in anns], dtype=torch.long)
            self.samples.append({'image_path': os.path.join(self.images_root, file_name), 'boxes': boxes, 'labels': labels})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['image_path']).convert('RGB')
        orig_w, orig_h = img.size
        img = img.resize((self.img_size, self.img_size))
        img_t = transforms.ToTensor()(img)
        # scale boxes to resized image
        if sample['boxes'].numel() == 0:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = sample['boxes'].clone()
            scale_x = self.img_size / orig_w
            scale_y = self.img_size / orig_h
            boxes[:, [0,2]] = boxes[:, [0,2]] * scale_x
            boxes[:, [1,3]] = boxes[:, [1,3]] * scale_y
            labels = sample['labels'].clone()
        target = {'boxes': boxes, 'labels': labels}
        if self.transform:
            img_t = self.transform(img_t)
        return img_t, target

def collate_fn(batch):
    imgs = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    imgs = torch.stack(imgs)
    return imgs, targets

# -------------------------
# Anchor Matching & Target Assignment
# -------------------------
def assign_targets_to_anchors(anchors, gt_boxes, gt_labels, positive_iou_thresh=0.5, negative_iou_thresh=0.4):
    """
    anchors: (N,4) tensor
    gt_boxes: (M,4)
    gt_labels: (M,)
    Returns:
      labels: (N,) with -1 ignore, 0 background, >0 class_id
      bbox_targets: (N,4) with encoded deltas for positive anchors (others zero)
    """
    N = anchors.shape[0]
    labels = torch.full((N,), -1, dtype=torch.long, device=anchors.device)  # default ignore
    bbox_targets = torch.zeros((N,4), dtype=torch.float32, device=anchors.device)

    if gt_boxes.numel() == 0:
        # no GT: all background
        labels[:] = 0
        return labels, bbox_targets

    ious = iou_xyxy(anchors, gt_boxes)  # (N, M)
    max_iou, argmax_iou = ious.max(dim=1)  # for each anchor
    # negative: iou < negative_iou_thresh
    labels[max_iou < negative_iou_thresh] = 0
    # positive: iou >= positive_iou_thresh
    pos_idx = max_iou >= positive_iou_thresh
    labels[pos_idx] = gt_labels[argmax_iou[pos_idx]]
    # for anchors with between thresh -> keep as -1 (ignore)
    # ensure each gt has at least one positive anchor (assign best anchor)
    best_anchor_iou_per_gt, best_anchor_idx = ious.max(dim=0)  # for each gt, best anchor index
    for gt_i in range(gt_boxes.shape[0]):
        anchor_idx = best_anchor_idx[gt_i]
        labels[anchor_idx] = gt_labels[gt_i]
        pos_idx[anchor_idx] = True
    # compute bbox targets for positive anchors
    pos_anchors = anchors[pos_idx]
    matched_gt = gt_boxes[argmax_iou[pos_idx]]
    if pos_anchors.shape[0] > 0:
        bbox_targets[pos_idx] = encode_boxes(pos_anchors, matched_gt)
    return labels, bbox_targets

# -------------------------
# Losses
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        """
        logits: (N, C) or (N,) for binary (we will assume single-class binary)
        targets: (N,) with 0..C labels (for multi-class we'd convert)
        For simplicity: we handle multi-class in one-vs-all manner outside if needed.
        In our usage, we will treat as binary per-anchor (object vs background) for single-class detection.
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_factor * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0/9, reduction='sum'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.beta, 0.5 * (diff ** 2) / self.beta, diff - 0.5 * self.beta)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# -------------------------
# Simple mAP@0.5 evaluator
# -------------------------
def nms(boxes, scores, iou_threshold=0.5):
    # boxes: (N,4), scores: (N,)
    if boxes.numel() == 0:
        return []
    keep = []
    idxs = scores.argsort(descending=True)
    while idxs.numel() > 0:
        i = idxs[0].item()
        keep.append(i)
        if idxs.numel() == 1:
            break
        ious = iou_xyxy(boxes[i:i+1], boxes[idxs[1:]]).squeeze(0)
        idxs = idxs[1:][ious <= iou_threshold]
    return keep

def compute_map(predictions, ground_truths, iou_thresh=0.5, num_classes=1):
    """
    predictions: list per image of dict {'boxes':Tensor(K,4), 'scores':Tensor(K), 'labels':Tensor(K)}
    ground_truths: list per image of dict {'boxes':Tensor(G,4), 'labels':Tensor(G)}
    Returns mAP@iou_thresh averaged over classes present in GT.
    This is a simple VOC2007-style AP calculation per class.
    """
    aps = []
    for cls in range(1, num_classes+1):
        # collect all predictions and gts of this class
        all_scores = []
        all_matches = []
        npos = 0
        gt_per_image = []
        for gt in ground_truths:
            mask = (gt['labels'] == cls)
            gt_boxes_cls = gt['boxes'][mask] if mask.any() else torch.zeros((0,4))
            gt_per_image.append({'boxes': gt_boxes_cls, 'detected': torch.zeros((gt_boxes_cls.shape[0],), dtype=torch.bool)})
            npos += gt_boxes_cls.shape[0]
        pred_list = []
        for img_i, pred in enumerate(predictions):
            mask = (pred['labels'] == cls)
            boxes = pred['boxes'][mask]
            scores = pred['scores'][mask]
            for b,s in zip(boxes, scores):
                pred_list.append({'image_idx': img_i, 'box': b, 'score': float(s)})
        if len(pred_list) == 0:
            aps.append(0.0)
            continue
        # sort preds by score desc
        pred_list = sorted(pred_list, key=lambda x: x['score'], reverse=True)
        tp = np.zeros((len(pred_list),), dtype=np.float32)
        fp = np.zeros((len(pred_list),), dtype=np.float32)
        for idx, p in enumerate(pred_list):
            img_idx = p['image_idx']
            box = p['box'].unsqueeze(0)
            gt_boxes = gt_per_image[img_idx]['boxes']
            if gt_boxes.shape[0] == 0:
                fp[idx] = 1
                continue
            ious = iou_xyxy(box.to(torch.float32), gt_boxes.to(torch.float32)).squeeze(0)
            max_iou, max_idx = ious.max(dim=0)
            if max_iou >= iou_thresh:
                if not gt_per_image[img_idx]['detected'][max_idx]:
                    tp[idx] = 1
                    gt_per_image[img_idx]['detected'][max_idx] = True
                else:
                    fp[idx] = 1
            else:
                fp[idx] = 1
        # compute precision-recall
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / max(1.0, npos)
        prec = tp_cum / (tp_cum + fp_cum + 1e-6)
        # AP: numerical integration (11-point or all-point). Use all-point
        ap = 0.0
        for t in np.linspace(0,1,101):
            p_at_t = prec[rec >= t].max() if np.any(rec >= t) else 0.0
            ap += p_at_t
        ap /= 101.0
        aps.append(ap)
    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps))

# -------------------------
# Training & Validation
# -------------------------
def prepare_targets_for_image(anchors, gt_boxes, gt_labels, device, pos_iou_thr, neg_iou_thr):
    labels, bbox_targets = assign_targets_to_anchors(anchors, gt_boxes.to(device), gt_labels.to(device),
                                                     positive_iou_thresh=pos_iou_thr, negative_iou_thresh=neg_iou_thr)
    return labels, bbox_targets

def model_outputs_to_detections(cls_outs, bbox_outs, anchors, image_size, score_thresh=0.05, nms_iou=0.5, max_detections=100):
    """
    cls_outs: list of tensors per level from model: (N, A*num_classes, H, W)
    bbox_outs: list of tensors per level: (N, A*4, H, W)
    anchors: list of anchors per level (tensors)
    Returns list per image dicts with boxes, scores, labels
    """
    device = cls_outs[0].device
    batch = cls_outs[0].shape[0]
    detections = []
    # flatten per level predictions
    all_boxes = []
    all_scores = []
    all_labels = []
    for lvl, (c, b, a) in enumerate(zip(cls_outs, bbox_outs, anchors)):
        N, C_mul, H, W = c.shape
        A = C_mul // 1  # assuming num_classes==1, adjust if multi-class
        # reshape: (N, A, C, H, W) but here C=1
        c = c.permute(0,2,3,1).reshape(N, -1)  # (N, K)
        b = b.permute(0,2,3,1).reshape(N, -1)  # (N, K*4)
        # each anchor has 4 bbox preds, need to reshape per anchor
        num_anchors_level = a.shape[0]
        # convert b to (N, num_anchors_level, 4)
        b = b.reshape(N, -1, 4)
        # convert c to (N, num_anchors_level)
        c = c.reshape(N, -1)
        all_boxes.append(b)
        all_scores.append(torch.sigmoid(c))
        # single-class labels = 1 for object
        all_labels.append(torch.ones_like(c, dtype=torch.long))
    # concat across levels
    all_boxes = torch.cat(all_boxes, dim=1)  # (N, total_anchors, 4)  <-- these are encoded deltas; need decode
    all_scores = torch.cat(all_scores, dim=1)  # (N, total_anchors)
    all_labels = torch.cat(all_labels, dim=1)
    # anchors concatenated needed to decode
    anchors_cat = torch.cat(anchors, dim=0).to(device)
    for i in range(batch):
        scores = all_scores[i]
        deltas = all_boxes[i]
        # decode deltas into boxes
        decoded = decode_boxes(anchors_cat, deltas)
        # clip to image
        decoded[:,0].clamp_(0, image_size-1)
        decoded[:,1].clamp_(0, image_size-1)
        decoded[:,2].clamp_(0, image_size-1)
        decoded[:,3].clamp_(0, image_size-1)
        # filter by score
        mask = scores > score_thresh
        if mask.sum() == 0:
            detections.append({'boxes': torch.zeros((0,4), device=device), 'scores': torch.zeros((0,), device=device),
                               'labels': torch.zeros((0,), dtype=torch.long, device=device)})
            continue
        boxes_f = decoded[mask]
        scores_f = scores[mask]
        labels_f = all_labels[i][mask]
        # simple class-agnostic NMS
        keep = nms(boxes_f, scores_f, iou_threshold=nms_iou)
        if len(keep) > 0:
            keep = keep[:max_detections]
            boxes_f = boxes_f[keep]
            scores_f = scores_f[keep]
            labels_f = labels_f[keep]
        detections.append({'boxes': boxes_f.detach().cpu(), 'scores': scores_f.detach().cpu(), 'labels': labels_f.detach().cpu()})
    return detections

def train_one_epoch(model, optimizer, dataloader, device, scaler, epoch, writer, cfg, anchors_all, anchors_per_level):
    model.train()
    running_cls_loss = 0.0
    running_reg_loss = 0.0
    focal_loss = FocalLoss(alpha=cfg['training'].get('focal_alpha', 0.25), gamma=cfg['training'].get('focal_gamma', 2.0), reduction='sum')
    reg_loss_fn = SmoothL1Loss(beta=1.0/9, reduction='sum')
    iters = len(dataloader)
    for i, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)
        batch_size = imgs.shape[0]
        # build per-image anchors list on device for model decoding and target assignment
        anchors_per_level_dev = [a.to(device) for a in anchors_per_level]
        anchors_all_dev = anchors_all.to(device)
        # forward
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=cfg['training'].get('amp', True)):
            cls_outs, bbox_outs = model(imgs)  # lists per level
            # prepare target tensors per image by matching anchors
            cls_losses = []
            reg_losses = []
            total_pos = 0
            for b_i in range(batch_size):
                gt_boxes = targets[b_i]['boxes']
                gt_labels = targets[b_i]['labels']
                labels, bbox_targets = assign_targets_to_anchors(anchors_all_dev, gt_boxes.to(device), gt_labels.to(device),
                                                                 positive_iou_thresh=cfg['training'].get('pos_iou', 0.5),
                                                                 negative_iou_thresh=cfg['training'].get('neg_iou', 0.4))
                # classification targets: objectness per-anchor (binary 0/1)
                cls_target = (labels > 0).float()
                # flatten model outputs across levels to match anchors_all ordering
                # flatten cls_outs & bbox_outs to (total_anchors,) and (total_anchors,4)
            # flattening across levels now (outside per-image loop)
            # Convert cls_outs list to tensors shaped (N, total_anchors)
            cls_logits_list = []
            bbox_preds_list = []
            for l_cls, l_box in zip(cls_outs, bbox_outs):
                # l_cls: (N, A*num_classes, H, W) ; l_box: (N, A*4, H, W)
                N, C, H, W = l_cls.shape
                cls_flat = l_cls.permute(0,2,3,1).reshape(N, -1)  # (N, K_l)
                box_flat = l_box.permute(0,2,3,1).reshape(N, -1, 4)  # (N, K_l, 4)
                cls_logits_list.append(cls_flat)
                bbox_preds_list.append(box_flat)
            cls_logits = torch.cat(cls_logits_list, dim=1)  # (N, total_anchors)
            bbox_preds = torch.cat(bbox_preds_list, dim=1)  # (N, total_anchors, 4)

            # Now compute loss per image using labels/bbox_targets computed earlier
            batch_cls_loss = 0.0
            batch_reg_loss = 0.0
            total_positive_anchors = 0
            for b_i in range(batch_size):
                gt_boxes = targets[b_i]['boxes']
                gt_labels = targets[b_i]['labels']
                labels, bbox_targets = assign_targets_to_anchors(anchors_all_dev, gt_boxes.to(device), gt_labels.to(device),
                                                                 positive_iou_thresh=cfg['training'].get('pos_iou', 0.5),
                                                                 negative_iou_thresh=cfg['training'].get('neg_iou', 0.4))
                cls_target = (labels > 0).float().to(device)  # (total_anchors,)
                cls_logit = cls_logits[b_i]  # (total_anchors,)
                bbox_pred = bbox_preds[b_i]  # (total_anchors,4)
                # classification loss: focal (binary)
                cls_loss = focal_loss(cls_logit, cls_target)  # summed
                # regression loss: only positives
                pos_mask = labels > 0
                num_pos = int(pos_mask.sum().item())
                total_positive_anchors += num_pos
                if num_pos > 0:
                    bbox_pred_pos = bbox_pred[pos_mask]
                    bbox_targets_pos = bbox_targets[pos_mask]
                    reg_loss = reg_loss_fn(bbox_pred_pos, bbox_targets_pos)
                else:
                    reg_loss = torch.tensor(0.0, device=device)
                batch_cls_loss += cls_loss
                batch_reg_loss += reg_loss
            # normalize losses by number of positive anchors (per standard practice)
            normalizer = max(1.0, float(total_positive_anchors))
            loss_cls = batch_cls_loss / normalizer
            loss_reg = batch_reg_loss / normalizer
            loss = loss_cls + cfg['training'].get('reg_weight', 1.0) * loss_reg

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_cls_loss += float(loss_cls.item())
        running_reg_loss += float(loss_reg.item())
        if i % cfg['training'].get('log_interval', 50) == 0:
            it = epoch * iters + i
            writer.add_scalar('train/loss_total', (running_cls_loss + running_reg_loss) / (i + 1e-9), it)
            writer.add_scalar('train/loss_cls', running_cls_loss / (i + 1e-9), it)
            writer.add_scalar('train/loss_reg', running_reg_loss / (i + 1e-9), it)
    # epoch summary
    return running_cls_loss / max(1, iters), running_reg_loss / max(1, iters)

def validate(model, dataloader, device, anchors_all, anchors_per_level, cfg):
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            cls_outs, bbox_outs = model(imgs)
            # get detections
            detections = model_outputs_to_detections(cls_outs, bbox_outs, anchors_per_level, cfg['training'].get('image_size',512),
                                                     score_thresh=cfg['training'].get('score_thresh',0.05),
                                                     nms_iou=cfg['training'].get('nms_iou',0.5),
                                                     max_detections=cfg['training'].get('max_dets',100))
            # append to lists (preds and gts per image)
            for i in range(len(detections)):
                pred = detections[i]
                preds.append(pred)
                gts.append({'boxes': targets[i]['boxes'], 'labels': targets[i]['labels']})
    num_classes = cfg['model'].get('num_classes', 1)
    map50 = compute_map(preds, gts, iou_thresh=0.5, num_classes=num_classes)
    return map50

# -------------------------
# Main
# -------------------------
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cfg = cfg['model']
    model = RetinaNet(backbone_pretrained=model_cfg.get('backbone_pretrained', True),
                      num_classes=model_cfg.get('num_classes', 1),
                      use_cbam=model_cfg.get('use_cbam', False),
                      fpn_out_channels=model_cfg.get('fpn_out_channels', 256))
    model.to(device)

    # prepare anchors (assume pyramid levels and strides compatible with backbone FPN)
    image_size = cfg['training'].get('image_size', 512)
    pyramid_levels = cfg['training'].get('pyramid_levels', [3,4,5,6])
    strides = cfg['training'].get('strides', [8,16,32,64])  # typical for P3..P6
    anchors_all, anchors_per_level = generate_pyramid_anchors(image_size=image_size,
                                                              pyramid_levels=pyramid_levels,
                                                              strides=strides,
                                                              base_size=cfg['anchors'].get('base_size', 16),
                                                              ratios=cfg['anchors'].get('ratios', [0.5,1.0,2.0]),
                                                              scales=cfg['anchors'].get('scales', [1.0, 1.2599, 1.5874]),
                                                              device=device)

    # dataset
    train_ann = cfg['training']['train_annotations']
    val_ann = cfg['training'].get('val_annotations', None)
    images_root = cfg['training'].get('images_root', '')
    transform = None
    train_ds = SmallObjectDataset(train_ann, images_root=images_root, img_size=image_size, transform=transform)
    val_ds = SmallObjectDataset(val_ann, images_root=images_root, img_size=image_size, transform=transform) if val_ann else None

    train_loader = DataLoader(train_ds, batch_size=cfg['training'].get('batch_size', 4),
                              shuffle=True, collate_fn=collate_fn, num_workers=cfg['training'].get('num_workers', 2))
    val_loader = DataLoader(val_ds, batch_size=cfg['training'].get('batch_size', 4),
                            shuffle=False, collate_fn=collate_fn, num_workers=cfg['training'].get('num_workers', 2)) if val_ds else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training'].get('lr', 1e-4),
                                 weight_decay=cfg['training'].get('weight_decay', 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training'].get('epochs', 20))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['training'].get('amp', True))

    log_dir = cfg['training'].get('log_dir', 'runs/exp_full')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    start_epoch = 0
    best_map = 0.0
    epochs = cfg['training'].get('epochs', 20)
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_cls_loss, train_reg_loss = train_one_epoch(model, optimizer, train_loader, device, scaler, epoch, writer, cfg, anchors_all, [a for a in anchors_per_level])
        scheduler.step()
        t1 = time.time()
        print(f"Epoch {epoch} train cls_loss={train_cls_loss:.4f} reg_loss={train_reg_loss:.4f} time={t1-t0:.1f}s")
        # validation
        if val_loader is not None:
            val_map = validate(model, val_loader, device, anchors_all, [a for a in anchors_per_level], cfg)
            print(f"Epoch {epoch} val mAP@0.5 = {val_map:.4f}")
            writer.add_scalar('val/map50', val_map, epoch)
            # save best
            if val_map > best_map:
                best_map = val_map
                ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'map50': val_map}
                torch.save(ckpt, os.path.join(log_dir, 'best_checkpoint.pth'))
        # regular checkpoint
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(log_dir, f'checkpoint_{epoch}.pth'))
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='02_model/configs/train_config.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)