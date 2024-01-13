$$\text{MMRMSE} = \frac{1}{R} \sum_{i=1}^R \left( \frac{1}{n} \sum_{j=1}^n(y_{ij} - \hat{y}_{ij})^2 \right)^{1/2}$$
- 𝑅 is the number of scored rows,
- $y_{ij}$ and $\hat{y}_{ij}$ are the actual and predicted values for row i and column j.
- n is the number of columns

Use when you have multiple targets to predict
- (which is why the submission is a matrix!)
Pros:
-  gives higher weight to larger errors (like RMSE)
