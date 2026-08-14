"""Potency and selectivity models.

``qsar`` trains RandomForest and XGBoost on the configured featurisation and reports conformal intervals;
``selectivity`` predicts the isoform potency difference directly, against a fixed potency-subtract
baseline it deliberately does not let configuration tune. Both are implemented and produce published
metrics.
"""
