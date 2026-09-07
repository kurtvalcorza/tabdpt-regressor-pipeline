# Tabular Regression Dataset Specification

## Required logical schema

A training table contains feature columns plus one regression target column.

- target name: configured by `target_column` (default `target`)
- target: numeric, finite, non-missing, and non-constant during fit
- duplicate column names: rejected
- feature columns: numeric, boolean, string, or categorical values supported
- empty feature set: rejected

## DIMER file layout

`DIMER_DATASET_DIR` must contain `train.csv` and may contain `val.csv`, either directly or within exactly one ZIP archive. If `val.csv` is absent, the runtime creates a deterministic holdout using `validation_split` and `seed` before applying the `max_train_rows` support cap.

## Preprocessing rules

1. Fit learned preprocessing only on the final training/support partition.
2. Numeric values are converted to floating point; missing/invalid numeric values become `NaN` and are handled by TabDPT's training-fitted preprocessing.
3. Categorical values are mapped using training-only vocabularies. Missing and unseen values have distinct codes.
4. The target is converted to `float64` before it is passed to TabDPT.
5. Configured `drop_columns` are removed before schema establishment and may be present in later raw tables.
6. After configured drop columns are removed, inference must contain exactly the fitted feature names. Missing or other extra columns are rejected; order is normalized to training order.

## Split rules

Random splitting is acceptable only for approximately IID rows. Preserve explicit partitions for temporal, grouped, panel, repeated-entity, patient, household, geographic, or other leakage-sensitive data.

## Operational envelope

The manifest defaults to at most 10,000 fitted support rows and a 2,048-row per-ensemble context. These are service controls rather than intrinsic TabDPT limits.

## Evaluation

Report MAE, RMSE, and R². Keep an independent test set whenever model or operational settings are selected using validation data. Do not report random-split metrics for forecasting-like data when chronology should define the split.
