**mean absolute percentage error**
link: 
range: 
### summary
$$
\begin{equation}
MAPE = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%
\end{equation}
$$
- n is the total number of data points
- This formula tells us: **How accurate the model is**
### pros
- it's scale-independent. so it's easy to compare between models
- works as a forecasting metric
- it gives more weight to data points with larger errors since it's percentage based
### Cons
- the absolute value bars makes it so we don't know if it's under or overestimating
- doesn't work if $y_i$ is 0