# Tabular Regression Dataset Specification

## Required logical schema

A training table contains feature columns plus one regression target column.

- target name: configured by `target_column` (default `target`)
- target: numeric, finite, non-missing, and non-constant
- duplicate column names: rejected
- feature columns: numeric, boolean, string, or categorical values supported
- empty feature set: rejected

## Preprocessing rules

1. Fit all learned preprocessing on the training partition only.
2. Numeric values are converted to floating point; missing/invalid numeric values become `NaN` and are handled by TabDPT's training-fitted imputer.
3. Categorical values are mapped using training-only vocabularies. Missing and unseen values have distinct codes.
4. The target is converted to `float64` before it is passed to TabDPT.
5. Inference tables must contain exactly the fitted feature names. Order is normalized to training order.

## Split rules

Random splitting is acceptable only for approximately IID rows. Preserve user-provided partitions for temporal, grouped, panel, repeated-entity, patient, household, geographic, or leakage-sensitive data.

## Operational envelope

The initial DIMER manifest defaults to at most 10,000 training rows and a prediction context of 2,048 rows. These are service controls, not intrinsic TabDPT model limits. Increase only after resource testing.

## Evaluation

Report MAE, RMSE, and R². Keep an independent test set whenever model or operational settings are selected using a validation partition. Do not report random-split metrics for forecasting-like data when chronology should define the split.