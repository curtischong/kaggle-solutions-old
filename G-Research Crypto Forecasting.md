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
		- there were around 200 weeks of data
		- used 6-fold, walk-forward, grouped cross validation
			- the group key was the timestamp
			- train folds were 40 weeks long
			- test folds were 40 weeks long
			- there was a gap of 1 week between test and train folds
			- the ends of training folds were incremented by 20 weeks each fold.
				- this sentence make me believe it's [[forward chaining cross validation]]
		- I chose to overlap my folds so that they could be long, but I could still have 6 folds.
			- more folds = less variance
			- I think he averaged these folds since they say "the average CV scores"
		- didn't change seeds cause the CV score variance was low enough
		- With no gap, a model can cheat at the the beginning of the test period, because the end of the train period is very similar.
			- obv, train without the gap for the final submission
	- feature engineering
		- they were very private about this because the competition host wouldn't make alpha
			- they definitely tried everything (e.g. making one model for all coints - the coin is a feature)
	- "Timeseries learning is essentially supervised learning that respects causality." (cause it all depends on past data)
	- models
		- LightGBM with squared loss
		- The only parameters I changed from defaults were the number of estimators, number of leaves, and the learning rate
		- no regularization, augmentation, or feature neutralization
			- it didn't help CV
		- I wanted to train with the entire dataset, because CV had shown me that scores just kept improving with longer training data
			- this is interesting. it suggests that limiting to only the last x weeks of data doesn't perform better than using all data (no domain drift)
		- df.update() caused lots of unnecessary RAM copies of arrays
			- The solution was to select small subsets of data, use df.update() on those subsets, then append to a list of dataframes, and then at the end, concatenate the list into one dataframe.
			- [[polars]] would've prob fixed this
			- he used [[Numba]] for feature engineering

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