# Breast Cancer Classification with Explainable AI

This project demonstrates a breast cancer classification model using **XGBoost**, enriched with **explainability techniques** via **SHAP** and **LIME**. It introduces a synthetic "age" feature to analyze model fairness and interpretability across different age groups.

---

## Model Overview

- **Model:** XGBoost Classifier
- **Dataset:** scikit-learn's built-in breast cancer dataset
- **Target:** Binary classification (Malignant vs Benign)
- **Added Feature:** Randomly generated `age` column (30–80)

---

## Explainability Methods

- **SHAP (SHapley Additive exPlanations):**
  - Summary plot
  - Dependence plot for a selected feature
  - SHAP value distribution across age groups

- **LIME (Local Interpretable Model-agnostic Explanations):**
  - (Prepared for use, not executed in the current script)

---

## Evaluation

- **Accuracy Score**
- **Classification Report**
- **SHAP-based feature importance**
- **Boxplot of SHAP values grouped by age**

---

## File Structure

```
breast_cancer_explainable_ai/
├── shap_lime_breast_cancer.py   # Main Python script
├── requirements.txt             # Required packages
└── README.md                    # Project overview (this file)
```

---

## How to Run

1. Clone the repository or download the files.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python shap_lime_breast_cancer.py
   ```

---

## Notes

- This project uses randomly generated age data for fairness analysis. It is **not medically accurate** and should be used for **educational purposes only**.
- `%matplotlib inline` was removed to ensure compatibility with `.py` files.

---

## Requirements

See `requirements.txt` or install manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap lime
```

---

## Future Improvements

- Integrate **LIME** local explanations into the script.
- Compare SHAP values between different subgroups (e.g., malignant vs benign).
- Extend fairness analysis to additional synthetic or real features.
