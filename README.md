# Logistic Regression — Airline Customer Satisfaction Prediction

## Dataset
**Invistico_Airline.csv** — 129,880 passenger survey records, 22 columns. Target variable: `satisfaction` (satisfied / dissatisfied). Features include customer type, age, type of travel, class, flight distance, 14 service-rating predictors (seat comfort, wifi, entertainment, etc.), and delay metrics. One column (`Arrival Delay in Minutes`) had 393 missing values, imputed with the median.

## Modeling Approach
1. **Cleaning**: imputed missing arrival-delay values with the median (robust to skew/outliers).
2. **Encoding**: target and two binary predictors (`Customer Type`, `Type of Travel`) mapped to 0/1; `Class` (3 categories) one-hot encoded with Eco as the reference level.
3. **Split**: 75/25 train/test split, stratified on the target to preserve class balance.
4. **Scaling**: features standardized (`StandardScaler`) before fitting, so coefficients are comparable in magnitude.
5. **Model**: binomial `LogisticRegression` (scikit-learn, `max_iter=1000`).

## Performance (held-out test set, n=32,470)

| Metric | Score |
|---|---|
| Accuracy | 0.829 |
| Precision | 0.843 |
| Recall | 0.844 |
| F1 | 0.844 |
| ROC-AUC | 0.903 |

**Confusion Matrix**

| | Predicted Dissatisfied | Predicted Satisfied |
|---|---|---|
| **Actual Dissatisfied** | 11,912 (TN) | 2,786 (FP) |
| **Actual Satisfied** | 2,776 (FN) | 14,996 (TP) |

The model performs well and fairly evenly across both classes — it correctly identifies ~81% of dissatisfied passengers and ~84% of satisfied ones, with a strong ROC-AUC of 0.90 indicating good overall class separation.

## Key Drivers of Satisfaction (standardized coefficients)

**Top positive drivers**: Inflight entertainment, Customer Type (loyalty), On-board service, Seat comfort, Checkin service.

**Top negative drivers (associated with dissatisfaction)**: Class (Eco / Eco Plus vs Business), Arrival delay, Food and drink, Departure/arrival time convenience.

## Business Recommendations
- **Invest in inflight digital experience** — entertainment and wifi are the strongest levers for moving dissatisfied passengers to satisfied.
- **Strengthen the online journey** — online boarding, support, and booking ease all matter; low-cost UX improvements can have outsized impact.
- **Improve cabin comfort and onboard service**, particularly for Eco/Eco Plus passengers who show systematically lower satisfaction than Business class.
- Loyal customers are far more likely to be satisfied — loyalty-program investment likely reinforces satisfaction (or vice versa); worth deeper causal analysis.

## Limitations & Next Steps
- Logistic regression captures linear (log-odds) relationships only; tree-based models (Random Forest, XGBoost) may capture interactions and lift recall further.
- Service ratings are correlated with each other (e.g., wifi and entertainment), so coefficients share some credit — standardized coefficients indicate relative, not strictly causal, importance.
- Before deployment, validate on more recent data and tune the classification threshold based on the airline's actual cost of false positives vs. false negatives for a retention campaign.

## Files
- `logistic_regression_analysis.ipynb` — full notebook, all cells executed, outputs visible.
- `Invistico_Airline.csv` — source dataset.
- `README.md` — this file.
- 
