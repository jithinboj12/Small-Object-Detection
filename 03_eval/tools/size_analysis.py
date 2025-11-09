import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
import matplotlib.pyplot as plt

def banded_mAP(cocoGt, cocoDt, imgIds=None, bands=[(0,32),(32,96),(96,1e9)]):
    """
    bands: list of (min_px, max_px) in pixels (area)
    returns dict of {band_label: AP}
    """
    res = {}
    for bmin, bmax in bands:
        ce = COCOeval(cocoGt, cocoDt, iouType='bbox')
        ce.params.imgIds = imgIds
        ce.params.areaRng = [[bmin, bmax]]
        ce.params.areaRngLbl = [f"{bmin}-{bmax}"]
        ce.evaluate(); ce.accumulate(); ce.summarize()
        res[f"{bmin}-{bmax}"] = ce.stats[0]
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--bands", nargs='*', default=["0-32","32-96","96-1000000000"])
    parser.add_argument("--img-ids", default=None)
    parser.add_argument("--plot", action='store_true')
    args = parser.parse_args()

    cocoGt = COCO(args.gt)
    cocoDt = cocoGt.loadRes(args.preds)
    imgIds = None
    if args.img_ids:
        with open(args.img_ids) as f:
            imgIds = json.load(f)

    bands_parsed = []
    for s in args.bands:
        a,b = s.split('-')
        bands_parsed.append((float(a), float(b)))
    res = banded_mAP(cocoGt, cocoDt, imgIds=imgIds, bands=bands_parsed)
    print("Banded AP results:")
    for k,v in res.items():
        print(k, v)

    if args.plot:
        xs = []
        ys = []
        for k,v in res.items():
            a = (float(k.split('-')[0]) + float(k.split('-')[1]))/2.0
            xs.append(a)
            ys.append(v)
        import matplotlib.pyplot as plt
        plt.plot(xs, ys, marker='o')
        plt.xlabel("object area (px^2)")
        plt.ylabel("AP")
        plt.title("AP vs object area bands")
        plt.grid(True)
        plt.savefig("ap_vs_size.png")
        print("Saved ap_vs_size.png")
