import json
import numpy as np
import argparse
from collections import Counter

def analyze_anchors(gt_json, top_k=6):
    with open(gt_json) as f:
        data = json.load(f)
    areas = []
    whs = []
    for ann in data['annotations']:
        w = ann['bbox'][2]
        h = ann['bbox'][3]
        areas.append(w*h)
        whs.append((w,h))
    areas = np.array(areas)
    print("Total boxes:", len(areas))
    print("Area stats: min %.1f median %.1f mean %.1f max %.1f" % (areas.min(), np.median(areas), areas.mean(), areas.max()))
    from sklearn.cluster import KMeans
    wh_arr = np.array(whs)
    kmeans = KMeans(n_clusters=top_k, random_state=0).fit(np.log(wh_arr+1))
    centers = np.exp(kmeans.cluster_centers_)-1
    centers = centers.astype(int).tolist()
    print("Suggested anchors (w,h):")
    for c in centers:
        print(c)
    for t in [16*16, 32*32, 64*64]:
        pct = np.mean(areas <= t)*100
        print(f"Boxes <= {t} px^2: {pct:.2f}%")
    return centers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()
    analyze_anchors(args.gt, top_k=args.k)
