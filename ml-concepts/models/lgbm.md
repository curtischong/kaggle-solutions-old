Light GBM

Params to tweak:
- TODO: add more params and add more opinions
- `boost_from_average=True`
	- https://github.com/Microsoft/LightGBM/issues/1913
	- `boost_from_average = True` means that the very first iteration will be a good default model that matches the base rate. As soon as you start boosting with `is_unbalance = True`, you're putting an unfair amount of weight on the minority class, such that a split will occur to reduce error on the minority class even if it disproportionally increases error on the majority class.
- `boosting='dart'` for [[D.A.R.T]]
- `max_depth`