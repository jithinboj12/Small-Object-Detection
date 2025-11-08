import numpy as np
import os
import config

def parse_dota_annotation(label_path):
    """
    Parses a DOTA annotation file.
    
    Returns:
        list: A list of dictionaries, each containing:
              - 'polygon': [x1, y1, x2, y2, x3, y3, x4, y4]
              - 'class': class name
              - 'difficult': difficulty flag (0 or 1)
    """
    objects = []
    if not os.path.exists(label_path):
        return []
        
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 10:
            continue
            
        objects.append({
            'polygon': [float(p) for p in parts[:8]],
            'class': parts[8],
            'difficult': int(parts[9])
        })
    return objects

def obb_to_hbb(polygon):
    """
    Converts an oriented bounding box (8 points) to a 
    horizontal bounding box (4 points: xmin, ymin, xmax, ymax).
    """
    pts = np.array(polygon).reshape(4, 2)
    xmin = np.min(pts[:, 0])
    ymin = np.min(pts[:, 1])
    xmax = np.max(pts[:, 0])
    ymax = np.max(pts[:, 1])
    return [xmin, ymin, xmax, ymax]

def hbb_to_yolo(hbb, class_id, img_width, img_height):
    """
    Converts a horizontal bounding box (xmin, ymin, xmax, ymax)
    to YOLO format (class_id, x_center_norm, y_center_norm, w_norm, h_norm).
    """
    xmin, ymin, xmax, ymax = hbb
    
    w = xmax - xmin
    h = ymax - ymin
    x_center = xmin + w / 2.0
    y_center = ymin + h / 2.0
    
    # Normalize
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    return [class_id, x_center_norm, y_center_norm, w_norm, h_norm]

def save_yolo_label(labels_list, output_path):
    """
    Saves a list of YOLO-formatted labels to a text file.
    Filters out any invalid labels.
    """
    with open(output_path, 'w') as f:
        for label in labels_list:
            # Basic validation: ensure all values are within [0, 1]
            if all(0.0 <= val <= 1.0 for val in label[1:]):
                class_id = int(label[0])
                vals = " ".join([f"{v:.6f}" for v in label[1:]])
                f.write(f"{class_id} {vals}\n")
            else:
                print(f"Warning: Skipping invalid label for {output_path}: {label}")