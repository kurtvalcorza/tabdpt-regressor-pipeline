# TabDPT Regressor Sample Datasets

This directory provides and documents sample datasets for the [TabDPT Regressor Colab Tutorials](../../tutorials/README.md).

TabDPT is an in-context learning tabular foundation model that consumes training rows as in-context support context (`training_context.csv`). The bundled sample dataset provides a reproducible, self-contained archive formatted for DIMER ingestion.

| Dataset | Modality / Task | Rows (Train / Val) | Features | Source & License | Archive SHA-256 |
|---|---|---|---|---|---|
| **Diabetes** | Quantitative regression | 353 train / 89 val (442 total) | 10 numeric | scikit-learn / Efron et al. (BSD 3-Clause) | `c6651628e61d298f2ebed8635f2217e49c5a3e59fbf4b3a9fee506dcb091bfec` |

---

## 1. Diabetes Progression

- **Purpose:** Quick smoke benchmark and DIMER contract validation. Pretrained TabDPT v1.2 produces quantitative disease progression predictions out of the box, verifying GPU/CPU execution, in-context conditioning, and artifact export.
- **Archive:** `diabetes.zip` containing `train.csv` and `val.csv`.
- **Target:** `target` — quantitative measure of disease progression one year after baseline.
- **Features:** 10 numeric physiological and blood serum attributes (age, sex, bmi, bp, s1, s2, s3, s4, s5, s6).
- **Split:** 80% train (353 rows) / 20% val (89 rows), random seed 42.
- **Specification Conformance:** Conforms strictly to [`TABULAR_REGRESSION_DATASET_SPEC.md`](../../TABULAR_REGRESSION_DATASET_SPEC.md). Contains non-missing, non-constant target column and finite numeric features.

---

## Deterministic Reproduction

The bundled archive is generated deterministically using [`examples/build_sample_datasets.py`](../build_sample_datasets.py):

```bash
python examples/build_sample_datasets.py
```
