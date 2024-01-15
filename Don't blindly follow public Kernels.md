
Everybody writes buggy code. Just because a kernel has many upvotes doesn't mean it's bug-free.

https://www.kaggle.com/code/awqatak/silver-bullet-single-model-165-features
```
rmse = mean_squared_error(y_true=y_val, y_pred=y_preds, squared=False)
# store the performance
weights.append(rmse)
```
- In this code, the rmse for each model is being used as the weights (for ensembling models)
- But if you notice:
	- rmse is an error! (the larger the value, the WORSE it is)
	- The weight determines the significance of the model.
- Here's the problem: models with a larger error is considered MORE in the final ensemble.
	- The rmse should've been raised to the power of -1 before being appended to `weights`.


As a competitor, we may not point out these bugs during the competition. But if you are using these kernels (after the competition)