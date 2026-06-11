import pickle
from pathlib import Path


pkl_path = Path("/root/foresee/FastWAM/debug_images/infer_kwargs_1781156490296.pkl")

with open(pkl_path, "rb") as f:
    infer_kwargs = pickle.load(f)

print(f"Keys: {list(infer_kwargs.keys())}\n")
for k, v in infer_kwargs.items():
    if hasattr(v, "shape"):
        print(f"{k}: type={type(v).__name__}, shape={v.shape}, dtype={getattr(v, 'dtype', None)}")
    elif isinstance(v, (list, tuple)):
        print(f"{k}: type={type(v).__name__}, len={len(v)}")
    else:
        print(f"{k}: type={type(v).__name__}, value={v}")
