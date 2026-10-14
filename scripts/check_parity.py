"""Parity check: the serving path must reproduce the offline pipeline."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.features as F        # noqa: E402
import src.pipeline as PL       # noqa: E402
import src.serving as SV        # noqa: E402

bundle = SV.load_bundle(ROOT / "artifacts")
clean = pd.read_parquet(ROOT / "data" / "processed" / "clean.parquet")
sample = clean.sample(2000, random_state=0).reset_index(drop=True)

# offline reference: the exact path used in Phase 4
offline = PL.predict(bundle["artifacts"], F.build_features(sample))

# A. serving path with the category column present: tests cleaning/feature reuse
served_a = SV.predict_orders(bundle, sample[SV.RAW_COLUMNS])
max_a = np.abs(served_a - offline).max()
print(f"A (category supplied): max |diff| = {max_a:.2e} s")

# B. API-style input: no category column, derived from store_id
raw_b = sample[SV.RAW_COLUMNS].drop(columns="store_primary_category")
diff_b = np.abs(SV.predict_orders(bundle, raw_b) - offline) / 60
print(f"B (category derived):  {(diff_b > 1e-9).mean():.1%} of rows differ | "
      f"mean |diff| = {diff_b.mean():.3f} min | max |diff| = {diff_b.max():.2f} min")

assert max_a < 1e-6, "serving path does not match the offline pipeline"
print("parity check passed")