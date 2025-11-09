# Small object evaluation & enhancement tools

Added tools in `tools/` for evaluation and model enhancement for small object detection.

1. Implement model adapter:
   - Edit `tools/adapter_template.py`: implement load_model() and predict(image).

2. Install extras:
   pip install -r tools/../requirements_extra.txt

3. Evaluate:
   python tools/eval_coco.py --coco-gt path/instances_val.json --images path/images --out preds.json

4. Size analysis:
   python tools/size_analysis.py --gt path/instances_val.json --preds preds.json --plot

5. Tiled inference:
   python tools/tile_inference.py --image path/large.jpg --tile-size 800 --overlap 0.2

6. Anchor tuning (if your model uses anchors):
   python tools/anchor_tuner.py --gt path/instances_train.json

7. Visualize:
   python tools/visualize_results.py --image path/img.jpg --out out.png
