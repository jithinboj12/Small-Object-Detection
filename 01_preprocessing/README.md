# 🛰️ Small Object Detection in Remote Sensing (Valise Technologies)

**Task 1: Dataset Collection & Preprocessing**  
**Author:** Manvith (Task 1)

---

This project takes the raw **DOTA dataset**, processes it efficiently into tiles, and prepares it for training a **small object detection model**.

This README is divided into two parts:

- **For the Team:** Instructions for Jithin, Gopica, and Rudra on how to use my output.  
- **For Developers:** Instructions on how to re-run my preprocessing (Task 1) from scratch.

---

## 🚀 For the Team (Jithin, Gopica, Rudra): Your Next Steps

My task (**Task 1**) is complete. I have processed the raw DOTA dataset and created a final, model-ready dataset for you.

### 📁 Your Files

The final deliverable is the `data/final_dataset` folder.  
I have zipped this folder as **`final_dataset.zip`** — this zip file is all you need.

---

### 📊 Dataset Details

- **Input Dataset:** DOTA v1.0  
- **Final Tile Size:** 1024x1024 (with a 200px overlap to catch objects on the edges)  
- **Label Format:** YOLO Format (`class_id, x_center_norm, y_center_norm, w_norm, h_norm`)  
- **Classes (15 total):**  
  plane, baseball-diamond, bridge, ground-track-field, small-vehicle, large-vehicle, ship, tennis-court, basketball-court, storage-tank, soccer-ball-field, roundabout, harbor, swimming-pool, helicopter

---

### 👩‍💻 Instructions for Each Teammate

#### 🧠 To Gopica (Task 3: Training Pipeline)

Your main file is `data/final_dataset/data.yaml`.

This one file tells your training script everything it needs to know:

- Path to the training images (`train: ...`)  
- Path to the validation images (`val: ...`)  
- Number of classes (`nc: 15`)  
- List of class names (`names: [...]`)

You just need to point your YOLO (or other) training script to this single `data.yaml` file.

---

#### ⚙️ To Rudra (Task 4: Evaluation)

You will use the `data/final_dataset/val` folder to perform evaluation.  
The `data.yaml` file also points to this validation set, so any scripts you write can (and should) read from that config file to get the path and class names.

---

#### 🧩 To Jithin (Task 2: Model Design)

Key information for you:

- **Input Resolution:** 1024x1024  
- **Problem:** This is a "small object detection" problem.  
- **Model Design Recommendations:**  
  - Choose a model (e.g., `YOLOv8-L`, `YOLOv8-X`) that can handle this resolution.  
  - Ensure P3/P4 feature maps are available for detecting small objects.  
  - Consider using BiFPN to enhance small-object detection.

The tiling process ensures small objects are not lost, so design your model accordingly.

---

## ⚙️ For Developers: How to Re-run My Preprocessing Work (Task 1)

This section explains how to re-run my code from scratch to regenerate the `final_dataset` folder.

---

### 🧩 Step 1: Setup Environment

(Only need to do this once)

```bash
# Create the virtual environment
python -m venv venv

# Activate it (Windows)
.venv\Scripts\Activate

# Install all required libraries
pip install -r requirements.txt
```

---

### 📦 Step 2: Download Raw Data (Manual Step)

This code does **not** download the data for you.

1. Go to the **DOTA Dataset Website**.  
2. Download **"Training set (images)"** and **"Training/Validation set (labelTxt)"**.  
3. Unzip and place the files in the correct folders:

```
data/raw/
├── images/
│   ├── P0001.png
│   └── ...
└── labelTxt/
    ├── P0001.txt
    └── ...
```

---

### 🧠 Step 3: Run the Efficient Tiling Script

This is the main, complex script. It uses multiprocessing (all your CPU cores) to process large images in parallel.

This will take a long time (e.g., 30–60 minutes).

```bash
python src/tile_processing.py
```

Output folder: `data/processed/`

---

### 🧾 Step 4: Create Final Train/Val Splits

This script is very fast. It takes all the tiles from `data/processed/` and splits them into the final `train/` and `val/` folders. It also creates the `data.yaml` file for the team.

```bash
python src/create_splits.py
```

Output folder: `data/final_dataset/`

---

### 🖼️ Step 5: Verify Your Work (Visual Check)

This is a utility script to visually check the output. It loads a random image from the final dataset and draws the bounding boxes to confirm label accuracy.

```bash
python src/explore.py
```

An image window will pop up. Press the **‘q’** key to close it.

---

### ✅ Summary

| Step | Description | Output Folder |
|------|--------------|----------------|
| 1 | Setup Environment | `venv/` |
| 2 | Download DOTA Dataset | `data/raw/` |
| 3 | Tile Processing | `data/processed/` |
| 4 | Create Train/Val Splits | `data/final_dataset/` |
| 5 | Visual Verification | (optional) |

---

### 📬 Author

**Manvith Kumar Reddy Devarapalli**  
Task 1 — Dataset Collection & Preprocessing  
*Small Object Detection in Remote Sensing (Valise Technologies)*
