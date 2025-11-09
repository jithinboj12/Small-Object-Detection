import cv2
import numpy as np
from adapter_template import load_model, predict

def draw_boxes(image, boxes, color=(255,0,0), label_prefix=''):
    im = image.copy()
    for b in boxes:
        x1,y1,x2,y2 = map(int,b['bbox'])
        score = b.get('score', None)
        cat = b.get('category_id', None)
        cv2.rectangle(im, (x1,y1), (x2,y2), color, 2)
        txt = f"{label_prefix}{cat}"
        if score is not None:
            txt += f" {score:.2f}"
        cv2.putText(im, txt, (x1, max(10,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return im

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="vis.png")
    args = parser.parse_args()
    load_model(device=args.device)
    import cv2
    im = cv2.imread(args.image)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    preds = predict(im_rgb, score_thresh=0.05)
    im_pred = draw_boxes(im_rgb, preds, color=(0,255,0), label_prefix='P:')
    im_out = cv2.cvtColor(im_pred, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.out, im_out)
    print("Wrote", args.out)
