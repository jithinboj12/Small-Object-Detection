import cv2
import os
import random
import albumentations as A
import config  # <-- THIS LINE IS NOW FIXED

def visualize_random_tile(dataset_split='train'):
    """
    Loads a random image and its label from the final dataset,
    draws the bounding boxes, and displays it.
    """
    print(f"Loading random tile from '{dataset_split}' split to verify...")
    
    # Define paths
    img_dir = os.path.join(config.FINAL_DATASET_DIR, dataset_split, 'images')
    lbl_dir = os.path.join(config.FINAL_DATASET_DIR, dataset_split, 'labels')
    
    # Pick a random image
    try:
        random_img_name = random.choice(os.listdir(img_dir))
    except IndexError:
        print(f"Error: No images found in {img_dir}.")
        print("Did you run 'python src/tile_processing.py' and 'python src/create_splits.py' first?")
        return
        
    img_path = os.path.join(img_dir, random_img_name)
    
    # Load image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Could not load image {img_path}")
        return
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    
    # Load corresponding label
    label_path = os.path.join(lbl_dir, os.path.splitext(random_img_name)[0] + '.txt')
    
    bboxes = []
    class_ids = []
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    class_id = int(parts[0])
                    
                    # Convert YOLO format back to (xmin, ymin, xmax, ymax)
                    x_c_norm, y_c_norm, w_norm, h_norm = map(float, parts[1:])
                    
                    box_w = w_norm * w
                    box_h = h_norm * h
                    xmin = (x_c_norm * w) - (box_w / 2)
                    ymin = (y_c_norm * h) - (box_h / 2)
                    xmax = xmin + box_w
                    ymax = ymin + box_h
                    
                    bboxes.append([xmin, ymin, xmax, ymax])
                    class_ids.append(class_id)
                except Exception as e:
                    print(f"Warning: Error parsing line '{line}': {e}")
                
    print(f"Found {len(bboxes)} objects in {random_img_name}")

    # --- Define Augmentation Pipeline (for verification) ---
    # This is also a good starting point for Gopica's task.
    # Remote sensing images can be rotated, so 'Rotate' is important.
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(p=0.3),
    ], bbox_params=A.BboxParams(format='pascal_voc', # (xmin, ymin, xmax, ymax)
                                label_fields=['class_labels']))

    # Apply transform
    try:
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_ids)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_labels = transformed['class_labels']
    except ValueError as e:
        print(f"Warning: Albumentations error (likely empty bboxes): {e}")
        print("Displaying original image instead.")
        transformed_image = image
        transformed_bboxes = bboxes
        transformed_labels = class_ids


    # --- Draw boxes on the (potentially augmented) image ---
    for bbox, class_id in zip(transformed_bboxes, transformed_labels):
        xmin, ymin, xmax, ymax = map(int, bbox)
        class_name = config.ID_TO_CLASS[class_id]
        
        # Draw rectangle (Green box)
        cv2.rectangle(transformed_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Put label
        cv2.putText(transformed_image, class_name, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
    # Display the image
    print("Displaying image. Press 'q' to quit.")
    transformed_image_bgr = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Verification - Press "q" to quit', transformed_image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_random_tile(dataset_split='train')