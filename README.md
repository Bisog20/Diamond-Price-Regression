# Diamond Price Prediction — Linear Regression

A baseline Linear Regression model predicting diamond prices from physical and quality attributes, with a focus on diagnosing and fixing a multicollinearity issue rather than just reporting a final metric.

## Dataset

53,940 diamonds with the following attributes:
- `carat` — weight of the diamond
- `cut` — quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- `color` — diamond color grade (J worst → D best)
- `clarity` — clarity grade (I1 worst → IF best)
- `depth`, `table` — proportion measurements
- `x`, `y`, `z` — physical dimensions (mm)
- `price` — target variable (USD)

## Data Cleaning

20 rows had impossible zero values in `x`, `y`, or `z` (a diamond can't have zero length/width/depth) and were dropped. `cut`, `color`, and `clarity` were ordinally encoded (not one-hot) since they represent graded scales, not unordered categories.

## Approach

1. **Baseline model** — Linear Regression on all 9 features
2. **Diagnosis** — residual plot revealed one prediction near **–$30,000**, a nonsensical result for a price that can't be negative
3. **Root cause** — `carat`, `x`, `y`, and `z` are all highly correlated (they all measure diamond size), causing multicollinearity and unstable coefficients
4. **Fix** — dropped `x`, `y`, `z` and refit using `carat` alone to represent size
5. **Result** — the fix improved both R² and RMSE despite using fewer features, and eliminated the outlier

## Results

| Metric | Baseline (9 features) | Final (6 features, no x/y/z) |
|---|---|---|
| R² | 0.9100 | **0.9112** |
| RMSE | $1,201.39 | **$1,193.23** |
| MAE | $790.37 | $839.65 |

Final model coefficients:

| Feature | Coefficient |
|---|---|
| carat | +8,768.93 |
| clarity | +519.07 |
| color | +318.10 |
| cut | +131.53 |
| table | –22.34 |
| depth | –40.69 |

All signs match domain intuition: larger, higher-clarity, higher-color, better-cut diamonds cost more.

## Known Limitation

The residual plot for the final model still shows a **funnel pattern** — residuals widen as predicted price increases. This means the model is less reliable for diamonds priced above ~$10,000; a production-grade pricing tool would likely need a log-transform of `price` or a tree-based model (Random Forest / Gradient Boosting) to handle this heteroscedasticity properly. That's a natural next step, not a fix applied here — this project is scoped as a Linear Regression baseline.

## Files

- `diamond_price_regression.ipynb` — full notebook, staged and commented
- `diamonds.csv` — dataset
- `residual_plot_baseline.png` — residual diagnostics before the fix
- `residual_plot_v2.png` — residual diagnostics after the fix
- `requirements.txt` — dependencies

## Run it yourself

```bash
pip install -r requirements.txt
jupyter notebook diamond_price_regression.ipynb
```
# Diamond-Price-Regression
