"""Assemble model inputs (features + leak-free store encoding) and model pipelines."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import src.encoders as E
import src.features as F
import src.split as S

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