"""Assemble model inputs (features + leak-free store encoding) and model pipelines."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import src.encoders as E
import src.features as F
import src.split as S
import lightgbm as lgb
import pandas as pd
STORE_FEATURES = ["store_te", "store_share"]
TIME_AS_CATEGORY = ["hour", "day_of_week"]       # non-monotonic: one-hot for linear models

RIDGE_NUMERIC = [c for c in F.NUMERIC_FEATURES if c not in TIME_AS_CATEGORY] + STORE_FEATURES
RIDGE_CATEGORICAL = F.CATEGORICAL_FEATURES + TIME_AS_CATEGORY


def add_store_features(train, others, smoothing=E.SMOOTHING):
    """Train rows are encoded out-of-fold; every frame in `others` uses an encoder fit on all of train."""
    y = train["duration"]
    groups = S.local_date(train["created_at"])
    encoded_train = train.join(E.oof_encode(train, y, groups, smoothing=smoothing))
    enc = E.StoreEncoder(smoothing).fit(train, y)
    return encoded_train, [o.join(enc.transform(o)) for o in others]


def build_ridge(alpha=10.0):
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50)
    prep = ColumnTransformer([("num", numeric, RIDGE_NUMERIC),
                              ("cat", categorical, RIDGE_CATEGORICAL)])
    return Pipeline([("prep", prep), ("model", Ridge(alpha=alpha))])

LGBM_NUMERIC = F.NUMERIC_FEATURES + STORE_FEATURES
LGBM_CATEGORICAL = F.CATEGORICAL_FEATURES

LGBM_PARAMS = dict(
    n_estimators=5000, learning_rate=0.05, num_leaves=63, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    cat_smooth=20, metric="l1", random_state=42, n_jobs=-1, verbose=-1,
)


def fit_categories(train):
    """Category levels seen in train; reused for validation, test and serving."""
    return {c: sorted(train[c].dropna().unique()) for c in LGBM_CATEGORICAL}


def to_lgbm_frame(df, categories):
    X = df[LGBM_NUMERIC + LGBM_CATEGORICAL].copy()
    for c in LGBM_CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=categories[c])   # unseen level -> NaN
    return X


def fit_lgbm(X_tr, y_tr, X_val, y_val, objective="regression", **overrides):
    params = {**LGBM_PARAMS, "objective": objective, **overrides}
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    return model