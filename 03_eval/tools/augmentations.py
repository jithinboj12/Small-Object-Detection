import random
import numpy as np
import cv2

def random_scale_jitter(image, boxes, scales=(0.8,1.2)):
    s = random.uniform(scales[0], scales[1])
    h,w = image.shape[:2]
    new_h = int(h*s)
    new_w = int(w*s)
    image = cv2.resize(image, (new_w, new_h))
    boxes = np.array(boxes, dtype=np.float32)
    boxes = boxes * s
    return image, boxes.tolist()

def random_crop_with_smallbox_focus(image, boxes, labels, crop_size=(512,512), tries=10):
    """
    Tries to crop region that includes small objects. Useful to create training patches.
    boxes: list of [x,y,w,h]
    labels: list parallel
    returns cropped_image, boxes', labels'
    """
    h,w = image.shape[:2]
    ch, cw = crop_size
    if w<=cw or h<=ch:
        return image, boxes, labels
    boxes_arr = np.array(boxes)
    areas = boxes_arr[:,2]*boxes_arr[:,3]
    small_idxs = np.where(areas <= (32*32))[0]
    if len(small_idxs)==0:
        x0 = random.randint(0, w-cw)
        y0 = random.randint(0, h-ch)
        crop = image[y0:y0+ch, x0:x0+cw]
        new_boxes=[]
        new_labels=[]
        for b,l in zip(boxes, labels):
            bx1,by1,bw,bh = b
            bx2 = bx1 + bw; by2=by1+bh
            if bx1 >= x0+cw or bx2 <= x0 or by1 >= y0+ch or by2 <= y0:
                continue
            nx1 = max(0, bx1-x0)
            ny1 = max(0, by1-y0)
            nx2 = min(cw, bx2-x0)
            ny2 = min(ch, by2-y0)
            nw = nx2-nx1; nh = ny2-ny1
            if nw<=0 or nh<=0:
                continue
            new_boxes.append([nx1, ny1, nw, nh])
            new_labels.append(l)
        return crop, new_boxes, new_labels
    for _ in range(tries):
        ri = int(random.choice(small_idxs))
        bx1,by1,bw,bh = boxes_arr[ri]
        cx = int(bx1 + bw/2)
        cy = int(by1 + bh/2)
        x0 = max(0, min(w-cw, cx - cw//2))
        y0 = max(0, min(h-ch, cy - ch//2))
        crop = image[y0:y0+ch, x0:x0+cw]
        new_boxes=[]
        new_labels=[]
        for b,l in zip(boxes, labels):
            bx1,by1,bw,bh = b
            bx2 = bx1 + bw; by2=by1+bh
            if bx1 >= x0+cw or bx2 <= x0 or by1 >= y0+ch or by2 <= y0:
                continue
            nx1 = max(0, bx1-x0)
            ny1 = max(0, by1-y0)
            nx2 = min(cw, bx2-x0)
            ny2 = min(ch, by2-y0)
            nw = nx2-nx1; nh = ny2-ny1
            if nw<=0 or nh<=0:
                continue
            new_boxes.append([nx1, ny1, nw, nh])
            new_labels.append(l)
        if len(new_boxes) > 0:
            return crop, new_boxes, new_labels
    return random_crop_with_smallbox_focus(image, boxes, labels, crop_size=(cw,cw), tries=1)
