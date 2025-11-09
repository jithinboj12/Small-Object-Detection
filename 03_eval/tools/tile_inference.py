import argparse
import os
import cv2
import numpy as np
from adapter_template import load_model, predict
from tqdm import tqdm

def nms(boxes, scores, iou_threshold=0.5):
    import numpy as np
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def tile_and_predict(image, tile_size=800, overlap=0.2, score_thresh=0.05):
    h,w,_ = image.shape
    stride = int(tile_size*(1-overlap))
    all_boxes = []
    for y in range(0, max(1, h - tile_size + 1), stride):
        for x in range(0, max(1, w - tile_size + 1), stride):
            tile = image[y:y+tile_size, x:x+tile_size]
            if tile.size == 0: 
                continue
            preds = predict(tile, score_thresh=score_thresh)
            for p in preds:
                bx = p['bbox']
                x1 = bx[0] + x
                y1 = bx[1] + y
                x2 = bx[2] + x
                y2 = bx[3] + y
                all_boxes.append({
                    'bbox': [x1,y1,x2,y2],
                    'score': p['score'],
                    'category_id': p['category_id']
                })
    final = []
    from collections import defaultdict
    by_cat = defaultdict(list)
    for i,b in enumerate(all_boxes):
        by_cat[b['category_id']].append((i,b))
    for cat, items in by_cat.items():
        boxes = [it[1]['bbox'] for it in items]
        scores = [it[1]['score'] for it in items]
        keep_idxs = nms(boxes, scores, iou_threshold=0.5)
        for k in keep_idxs:
            final.append(items[k][1])
    return final

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--tile-size", type=int, default=800)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    load_model(device=args.device)
    im = cv2.imread(args.image)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    preds = tile_and_predict(im, tile_size=args.tile_size, overlap=args.overlap)
    print(f"Found {len(preds)} detections after tiling")
    # optional print
    for p in preds[:50]:
        print(p)
