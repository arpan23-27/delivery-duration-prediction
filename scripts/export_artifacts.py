"""Copy the Phase 4 artifacts into the tracked artifacts/ folder and write metadata."""
import json
import platform
import shutil
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)

shutil.copy(ROOT / "models" / "final.joblib", OUT / "final.joblib")
shutil.copy(ROOT / "data" / "processed" / "store_category_map.json",
            OUT / "store_category_map.json")

metrics = json.loads((ROOT / "models" / "test_metrics.json").read_text())
final_row = {m: v["lightgbm huber (final)"] for m, v in metrics.items()}   # columns orientation

metadata = {
    "model_version": "0.6.0",
    "model": "LightGBM, Huber objective (alpha = 600 s)",
    "trained_on": "train+val: orders created 2015-01-21 to 2015-02-13 (Pacific days)",
    "test_metrics": final_row,
    "python": platform.python_version(),
    "libraries": {p: version(p) for p in
                  ["pandas", "numpy", "scikit-learn", "lightgbm", "joblib"]},
}
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata, indent=2))