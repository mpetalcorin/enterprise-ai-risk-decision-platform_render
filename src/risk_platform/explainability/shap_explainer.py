from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from risk_platform.features.pipeline import FEATURE_COLUMNS


class Explainer:
    """Lightweight Tree-SHAP explanations using XGBoost's native pred_contribs.

    XGBoost computes exact Tree-SHAP contribution values internally, so the
    free web-demo runtime does not need to import the heavier ``shap`` package.
    The full development environment can still use SHAP elsewhere in the repo.
    """

    def __init__(self, predictor):
        self.predictor = predictor
        self._booster = None
        if predictor.backend == "xgboost":
            try:
                self._booster = predictor.bundle["estimator"].get_booster()
            except Exception:
                self._booster = None

    def top_drivers(self, raw: pd.DataFrame, n: int = 5) -> list[list[dict[str, float | str]]]:
        features = self.predictor.feature_frame(raw)
        if self._booster is None:
            return [[] for _ in range(len(features))]

        matrix = xgb.DMatrix(features, feature_names=FEATURE_COLUMNS)
        # pred_contribs returns Tree-SHAP contributions plus a final bias term.
        values = self._booster.predict(matrix, pred_contribs=True)[:, :-1]
        result: list[list[dict[str, float | str]]] = []
        for row in np.asarray(values):
            idx = np.argsort(np.abs(row))[::-1][:n]
            result.append(
                [
                    {
                        "feature": FEATURE_COLUMNS[i],
                        "contribution": float(row[i]),
                        "direction": "increases_risk" if row[i] >= 0 else "decreases_risk",
                    }
                    for i in idx
                ]
            )
        return result
