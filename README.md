# Loan Approval Model for FinTech Innovations

A CRISP-DM machine learning project predicting historical loan approval decisions
for a fintech risk analytics team, with a cost-aware decision threshold and an
interpretable final model.

## Summary

A tuned logistic regression predicts loan approval with **ROC-AUC 0.995** and
**PR-AUC 0.984** on a held-out test set. Using a decision threshold derived from
the business's stated costs (false approval $50,000, false denial $8,000), the
model costs about **$602 per applicant** versus $1,912 for denying every
applicant and $38,050 for approving every applicant — a roughly 68% reduction
in expected cost compared to the best naive policy.

## Repository contents

| File | Description |
|---|---|
| `financial_loan_risk_completed.ipynb` | Full analysis: business understanding, EDA, preprocessing, modeling, tuning, evaluation, and business recommendations |
| `financial_loan_data.csv` | Applicant-level loan data (20,000 rows, 35 columns) |
| `requirements.txt` | Python dependencies |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook financial_loan_risk_completed.ipynb
```

## Approach

- **Framing:** Classification on `LoanApproved`, chosen because the business
  costs attach to an approve/deny decision (see notebook for full reasoning).
- **Leakage:** `RiskScore` is excluded — it is computed from the approval
  decision itself in the underlying data-generation process.
- **Preprocessing:** A `ColumnTransformer` + `Pipeline` with separate flows for
  numeric, ordinal (`EducationLevel`), and categorical features.
- **Models compared:** Logistic Regression, Random Forest, Histogram Gradient
  Boosting — tuned with `GridSearchCV` / `RandomizedSearchCV`.
- **Final model:** Logistic Regression, selected for its top cross-validated
  performance, speed, and interpretability for regulatory purposes.
- **Decision threshold:** Set from the business's cost ratio rather than the
  default 0.5, using out-of-fold training predictions only.

## Key limitations

- The target reflects historical *decisions*, not actual loan repayment
  outcomes, so the model reproduces any bias in past decisions.
- The dataset is synthetic and rule-generated, which explains the unusually
  high scores.
- `Age` and `MaritalStatus` are used as model inputs; a production deployment
  would need compliance review of fair-lending implications before launch.

Full details, including EDA findings, hyperparameter search results, feature
importance, and business recommendations, are in the notebook.
