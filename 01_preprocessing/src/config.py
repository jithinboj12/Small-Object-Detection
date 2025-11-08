import os

# --- 1. Main Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
RAW_IMAGE_DIR = os.path.join(RAW_DATA_DIR, 'images')
RAW_LABEL_DIR = os.path.join(RAW_DATA_DIR, 'labelTxt')

PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
PROCESSED_IMAGE_DIR = os.path.join(PROCESSED_DATA_DIR, 'images')
PROCESSED_LABEL_DIR = os.path.join(PROCESSED_DATA_DIR, 'labels')

FINAL_DATASET_DIR = os.path.join(DATA_DIR, 'final_dataset')

# --- 2. DOTA Class List ---
# These are the 15 classes from DOTA v1.0
CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field',
    'roundabout', 'harbor', 'swimming-pool', 'helicopter'
]

# Create a mapping from class name to an integer ID
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
ID_TO_CLASS = {i: name for i, name in enumerate(CLASSES)}

# --- 3. Tiling Parameters ---
# We will use 1024x1024 tiles with a 200px overlap
TILE_SIZE = 1024
TILE_OVERLAP = 200

# --- 4. Dataset Split Parameters ---
VALIDATION_SPLIT_PERCENT = 0.2
RANDOM_SEED = 42