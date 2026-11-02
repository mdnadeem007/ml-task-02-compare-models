# Task 02: Train and Compare Models

Three different algorithms trained on the same problem, same split, and compared honestly using a metric that fits the problem.

## Problem and dataset

**Breast Cancer Wisconsin (Diagnostic)**, bundled with scikit-learn (`load_breast_cancer`), so nothing needs downloading.

- 569 rows, 30 numeric features
- Binary target: malignant (0) or benign (1)

## What the notebook does

1. Splits the data **once** (80/20, stratified, `random_state=42`) and uses that same split for every model
2. Trains three algorithms from different families, plus the trivial baseline from Task 01:
   - Logistic Regression (scaled)
   - K-Nearest Neighbours, k=5 (scaled)
   - Random Forest, 200 trees
3. Compares them in one table
4. Runs 5-fold cross-validation on the training data as a stability check

## Metric choice

Accuracy treats all errors equally, but missing a malignant tumour is worse than a false alarm. The main metric is therefore **recall for the malignant class**. Precision, F1 and accuracy are also shown. Malignant is class `0` in this dataset, so `pos_label=0` is used throughout.

## Results (test set, 114 rows)

| Model | Recall (malignant) | Precision (malignant) | F1 (malignant) | Accuracy | CV recall (mean ± std) |
|---|---|---|---|---|---|
| Baseline (most frequent) | 0.000 | 0.000 | 0.000 | 0.632 | 0.000 ± 0.000 |
| Logistic Regression | 0.976 | 0.976 | 0.976 | 0.982 | 0.965 ± 0.029 |
| KNN (k=5) | 0.929 | 0.951 | 0.940 | 0.956 | 0.929 ± 0.024 |
| Random Forest | 0.929 | 0.951 | 0.940 | 0.956 | 0.953 ± 0.030 |

## Conclusion

All three models beat the baseline by a wide margin. The baseline reaches 63% accuracy yet catches no malignant tumours, which shows why accuracy alone is misleading here. Logistic Regression scored best on the test set, but the gaps are only 1 to 2 test samples and are about the size of the cross-validation spread, so the honest conclusion is that all three perform well. Logistic Regression is a reasonable choice because it is the simplest and scored best.

## Decisions and limitations

- Same split for all models, so the comparison is fair.
- No hyperparameter tuning, so no model gets an advantage from extra effort.
- Scalers sit inside pipelines so test data never influences training.
- The test set is small (114 rows); results are approximate.

## How to run

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
jupyter notebook compare_models.ipynb
```

Then choose **Kernel > Restart & Run All**. Or upload `compare_models.ipynb` to Google Colab and run all cells.

## Files

- `compare_models.ipynb`: the notebook
- `requirements.txt`: dependencies
- `README.md`: this file
