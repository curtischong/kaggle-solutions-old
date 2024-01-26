https://medium.com/@meir412_37692/d-a-r-t-your-new-weapon-against-overfitting-in-boosting-models-9ea4e6aa435b
- Dropouts meet Multiple Additive Regression Trees
- In D.A.R.T, the meaning of dropout is to ignore part of the trees when calculating the pseudo-residuals in each iteration.
	- D.A.R.T requires an additional hyperparameter, which determines the proportion of trees that **should stay for the calculation of the pseudo-residuals in each iteration**.
- this is a form of [[regularization]]