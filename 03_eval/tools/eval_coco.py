import argparse
import json
import os
from pathlib import Path
import numpy as np
from adapter_template import load_model, predict

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception as e:
    raise ImportError("pycocotools not installed. pip install pycocotools") from e

def convert_preds_to_coco_format(image_id, preds):
    coco_preds = []
    for p in preds:
        x1,y1,x2,y2 = p['bbox']
        w = x2 - x1
        h = y2 - y1
        coco_preds.append({
            "image_id": int(image_id),
            "category_id": int(p['category_id']),
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "score": float(p['score'])
        })
    return coco_preds

def run_eval(coco_gt_json, images_dir, out_json="preds.json", device='cuda', score_thresh=0.05):
    load_model(device=device)
    cocoGt = COCO(coco_gt_json)
    imgIds = cocoGt.getImgIds()
    all_preds = []
    for img_id in imgIds:
        img_info = cocoGt.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        img_path = os.path.join(images_dir, file_name)
        if not os.path.exists(img_path):
            print(f"Warning: image {img_path} not found — skipping")
            continue
        # load image with cv2
        import cv2
        im = cv2.imread(img_path)
        if im is None:
            print(f"Warning: couldn't read {img_path}")
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        preds = predict(im, score_thresh=score_thresh)
        coco_preds = convert_preds_to_coco_format(img_id, preds)
        all_preds.extend(coco_preds)

    with open(out_json, 'w') as f:
        json.dump(all_preds, f)
    print(f"Wrote {len(all_preds)} predictions to {out_json}")

    cocoDt = cocoGt.loadRes(out_json)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()  # prints AP @[IoU=0.50:0.95] and size breakdown
    stats = cocoEval.stats  # array of metrics
    metrics = {
        'AP (IoU=0.50:0.95)': stats[0],
        'AP@0.5': stats[1],
        'AP@0.75': stats[2],
        'AP (small)': stats[3],
        'AP (medium)': stats[4],
        'AP (large)': stats[5],
        'AR@1': stats[6],
        'AR@10': stats[7],
        'AR@100': stats[8],
        'AR (small)': stats[9],
        'AR (medium)': stats[10],
        'AR (large)': stats[11],
    }
    print("\nDetailed metrics:")
    for k,v in metrics.items():
        print(f"{k:20s}: {v:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coco-gt", required=True, help="COCO style ground truth JSON")
    p.add_argument("--images", required=True, help="Directory with images referenced by GT")
    p.add_argument("--out", default="preds.json", help="Output COCO preds JSON")
    p.add_argument("--device", default="cuda")
    p.add_argument("--score-thresh", default=0.05, type=float)
    args = p.parse_args()
    run_eval(args.coco_gt, args.images, out_json=args.out, device=args.device, score_thresh=args.score_thresh)
