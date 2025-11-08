import os
import cv2
import numpy as np
import config, utils  # <-- THIS LINE IS NOW FIXED
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_image(image_filename):
    """
    This function is the core of the preprocessing pipeline.
    It's designed to be run in a separate process.
    
    It takes one large image, tiles it, and saves all corresponding
    image tiles and label files.
    """
    try:
        # 1. --- Load Image and Label ---
        image_path = os.path.join(config.RAW_IMAGE_DIR, image_filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return 0  # Return 0 tiles processed
            
        img_h, img_w, _ = img.shape
        
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(config.RAW_LABEL_DIR, label_filename)
        
        raw_objects = utils.parse_dota_annotation(label_path)
        
        # Convert all OBBs to HBBs for easier processing
        # We store them as (class_name, [xmin, ymin, xmax, ymax])
        all_hbbs = []
        for obj in raw_objects:
            if obj['class'] in config.CLASS_TO_ID:
                all_hbbs.append({
                    'class_name': obj['class'],
                    'hbb': utils.obb_to_hbb(obj['polygon'])
                })
        
        if not all_hbbs:
            return 0 # No objects to process in this image

        # 2. --- Generate Tile Coordinates ---
        step = config.TILE_SIZE - config.TILE_OVERLAP
        tiles_processed = 0
        
        for y in range(0, img_h, step):
            for x in range(0, img_w, step):
                # Define tile boundaries
                x_min_tile, y_min_tile = x, y
                x_max_tile = min(x + config.TILE_SIZE, img_w)
                y_max_tile = min(y + config.TILE_SIZE, img_h)
                
                # Skip tiles that are too small (e.g., slivers at the edge)
                if (x_max_tile - x_min_tile) < config.TILE_OVERLAP or \
                   (y_max_tile - y_min_tile) < config.TILE_OVERLAP:
                    continue

                # 3. --- Find Objects in This Tile ---
                tile_labels_yolo = []
                
                for obj in all_hbbs:
                    xmin, ymin, xmax, ymax = obj['hbb']
                    
                    # Check if the center of the object is in the tile
                    # This is a common way to handle objects at tile boundaries
                    obj_center_x = (xmin + xmax) / 2
                    obj_center_y = (ymin + ymax) / 2
                    
                    if (x_min_tile <= obj_center_x < x_max_tile) and \
                       (y_min_tile <= obj_center_y < y_max_tile):
                        
                        # Translate object coords to be *relative* to the tile
                        new_xmin = max(0, xmin - x_min_tile)
                        new_ymin = max(0, ymin - y_min_tile)
                        new_xmax = min(config.TILE_SIZE, xmax - x_min_tile)
                        new_ymax = min(config.TILE_SIZE, ymax - y_min_tile)
                        
                        # Get tile dimensions
                        tile_w = x_max_tile - x_min_tile
                        tile_h = y_max_tile - y_min_tile
                        
                        # Get class ID
                        class_id = config.CLASS_TO_ID[obj['class_name']]
                        
                        # Convert to YOLO format
                        yolo_box = utils.hbb_to_yolo(
                            [new_xmin, new_ymin, new_xmax, new_ymax],
                            class_id,
                            tile_w,
                            tile_h
                        )
                        tile_labels_yolo.append(yolo_box)

                # 4. --- Save Tile and Labels ---
                # We only save tiles that contain at least one object
                if tile_labels_yolo:
                    # Extract image tile
                    tile_img = img[y_min_tile:y_max_tile, x_min_tile:x_max_tile]
                    
                    # Define unique tile filename
                    base_filename = os.path.splitext(image_filename)[0]
                    tile_filename = f"{base_filename}__{y_min_tile}_{x_min_tile}.png"
                    
                    # Save image tile
                    output_img_path = os.path.join(config.PROCESSED_IMAGE_DIR, tile_filename)
                    cv2.imwrite(output_img_path, tile_img)
                    
                    # Save label tile
                    output_label_path = os.path.join(config.PROCESSED_LABEL_DIR, 
                                                     f"{base_filename}__{y_min_tile}_{x_min_tile}.txt")
                    utils.save_yolo_label(tile_labels_yolo, output_label_path)
                    
                    tiles_processed += 1
                    
        return tiles_processed
        
    except Exception as e:
        print(f"Error processing {image_filename}: {e}")
        return 0

def run_tiling():
    """
    Main function to run the parallel processing.
    """
    print("Starting efficient tiling process...")
    
    # --- 1. Create output directories ---
    os.makedirs(config.PROCESSED_IMAGE_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_LABEL_DIR, exist_ok=True)
    
    # --- 2. Get list of images to process ---
    image_files = [f for f in os.listdir(config.RAW_IMAGE_DIR) 
                   if f.endswith(('.png', '.jpg', '.tif'))]
    
    if not image_files:
        print(f"Error: No images found in {config.RAW_IMAGE_DIR}")
        return

    print(f"Found {len(image_files)} large images to process.")

    # --- 3. Run in Parallel ---
    # Use ProcessPoolExecutor for CPU-bound tasks (image processing)
    # We leave one CPU free for the OS
    num_workers = max(1, os.cpu_count() - 1)
    print(f"Using {num_workers} parallel workers.")
    
    total_tiles_created = 0
    
    # This is the "efficient" part. It processes images in parallel.
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Create a "future" for each image processing task
        futures = {executor.submit(process_single_image, img_file): img_file 
                   for img_file in image_files}
        
        # Use tqdm to show a progress bar
        for future in tqdm(as_completed(futures), total=len(image_files), desc="Processing Images"):
            img_file = futures[future]
            try:
                tiles_created = future.result()
                total_tiles_created += tiles_created
            except Exception as e:
                print(f"An error occurred while processing {img_file}: {e}")

    print("\n--- Tiling Complete ---")
    print(f"Processed {len(image_files)} large images.")
    print(f"Created {total_tiles_created} small (tiled) image/label pairs.")

if __name__ == "__main__":
    # This allows you to run: python src/tile_processing.py
    run_tiling()