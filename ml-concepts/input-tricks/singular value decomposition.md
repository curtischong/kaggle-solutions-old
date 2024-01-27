- It's a technique to denoise target variables
	- https://www.kaggle.com/code/ambrosm/scp-quickstart?scriptVersionId=144293041&cellId=8
	- kinda like log transforming y in linear regression to make the target variable have constant variance
		- except this is for matrices of values
- Since SVD is applied to a matrix, only use this when you are predicting on multiple target variables
- PLEASE REMEMBER TO INVERSE TRANSFORM YHAT WHEN INFERENCING

e.g.
```python
svd = TruncatedSVD(n_components=n_components, random_state=1)

# fit the svd and your model
z_tr = svd.fit_transform(train[genes])
model.fit(train[features], z_tr)

# look! we're inverse transforming before returning the preds!
y_pred = svd.inverse_transform(model.predict(val[features]))
```