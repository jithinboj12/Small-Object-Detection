import numpy as np
import torch
_model = None
_device = None
def load_model(device='cuda'):
    global _model, _device
    _device = torch.device(device if torch.cuda.is_available() else 'cpu')
    raise NotImplementedError("Edit tools/adapter_template.py to load your model")
def predict(image: np.ndarray, score_thresh=0.05):
    """
    image: HxWxC, uint8 RGB (0-255) or BGR (if your model expects CV2 BGR).
    returns: list of dicts with keys 'bbox', 'score', 'category_id'
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    raise NotImplementedError("Edit predict() to call your model and return standardized outputs")
