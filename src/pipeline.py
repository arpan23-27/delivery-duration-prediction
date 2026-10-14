"""Final pipeline: fit on given rows, predict from engineered features, save/load artifacts."""
from pathlib import Path

import joblib
import lightgbm as lgb

import src.encoders as E
import src.modeling as MD

FINAL_OBJECTIVE = "huber"
FINAL_ALPHA = 600
TREE_SCALE = 1.2      # final fit has ~20% more rows than the fit that chose the tree count


def fit_final(fit_df, n_trees, smoothing=E.SMOOTHING):
    """fit_df: engineered features plus `duration`. Encoder, categories and model fit on it only."""
    y = fit_df["duration"]
    encoded, _ = MD.add_store_features(fit_df, [], smoothing)      # out-of-fold for these rows
    encoder = E.StoreEncoder(smoothing).fit(fit_df, y)              # used for any new rows
    categories = MD.fit_categories(fit_df)
    X = MD.to_lgbm_frame(encoded, categories)
    params = {**MD.LGBM_PARAMS, "objective": FINAL_OBJECTIVE,
              "alpha": FINAL_ALPHA, "n_estimators": n_trees}
    model = lgb.LGBMRegressor(**params).fit(X, y)
    return {"model": model, "encoder": encoder, "categories": categories}


def predict(artifacts, feats):
    """feats: output of build_features. This is the exact path the API will use."""
    enc = feats.join(artifacts["encoder"].transform(feats))
    X = MD.to_lgbm_frame(enc, artifacts["categories"])
    return artifacts["model"].predict(X)


def save_artifacts(artifacts, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, path)


def load_artifacts(path):
    return joblib.load(path)