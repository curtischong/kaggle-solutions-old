**Link:** https://www.kaggle.com/c/g-research-crypto-forecasting
**Problem Type:** [[Time Series]]
**Input:** the price, volume, OHLC and num trades in that time interval (a minute)
- NOTE: you do not get the candlestick charts. You only get aggregate time series data for each minute
**Output:** 15 minute residualized returns
**Eval Metric:**
Your (forecasted returns) - (actual market returns)
- here the returns are logged
	- it's calculated by: log(price in current time step / price in previous time step)
- the market returns are the weighted sums of each asset's returns (each asset gets a diff weight)
- more details here: https://www.kaggle.com/code/cstein06/tutorial-to-the-g-research-crypto-competition/notebook
##### Summary
##### Solutions
- (2nd)
	- https://www.kaggle.com/competitions/g-research-crypto-forecasting/discussion/323098
	- "I'll strive to provide insight into my methods, but without giving away model details that could be profitable for the host"
	- ### Crossvalidation
		- used 6-fold, walk-forward, grouped cross validation
			- the group key was the timestamp
			- train folds were 40 weeks long
			- test folds were 40 weeks long
			- there was a gap of 1 week between test and train folds
			- the ends of training folds were incremented by 20 weeks each fold.
		- I chose to overlap my folds so that they could be long, but I could still have 6 folds. With non-overlapping folds, I would have to decide between many short folds or a few long ones.

##### Important notebooks/discussions
- https://www.kaggle.com/code/cstein06/tutorial-to-the-g-research-crypto-competition/notebook
	-  log returns are preferred for mathematical modelling of time series
		- cause they are additive across time
		- and they are not bounded (regular returns cannot go below -100%)
		- you just log the return to calculate this (diff between curr price and the prev price)
	- there is high but variable correlation between the price of BTC and ETH
	- we want to work with constant mean, variance, and autocorrelation variables
		- since these are stationary distributions, and we can use forecasting techniques on it
	- use StandardScaler and fit_transform your input/output feats (cause regression models care about being in the proper scale)


#### Takeaways