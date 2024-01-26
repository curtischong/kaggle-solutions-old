**Link:** https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting
**Problem Type:** [[Time Series]]
**Input:** game statistics for the current day
**Output:** a prediction of how engaged fans are with a specific player after a game
- there are 4 target columns, each representing the engagement level fans have with that player
- we are predicting the engagement levels 1 day after the game (i.e. 1 day after the x_train data)
	- however, during the evaluation period, "The test data arrives in a data frame identical in format to **train.csv**, except it does not contain the target values."
	- so even though you got those engagement levels, you can't use today's engagement level to predict tomorrow's engagement level. Since you don't know what today's engagement level is!
**Eval Metric:** [[mean column-wise mean absolute error (MCMAE)]]
##### Summary

- Note: 
	- Binary columns will have null values as well as zeroes. Zeroes will occur if a player had an opportunity to do something, but did not. Nulls will occur if a player never had the opportunity to do something
		- e.g. a player who does not pitch on a given day cannot possibly pitch a shutout
		- how did they solve this?
##### Solutions
- (1st) lots of model testing, not much features
	- https://www.kaggle.com/competitions/mlb-player-digital-engagement-forecasting/discussion/274255
	- solution code: https://www.kaggle.com/code/ph0921/mlb-final-1/notebook
	- [[log scaling]]
		- all the past targets were scaled with log(1+x) to reduce their partly extremely "peaked nature"
		- in addition a target mask was added to mark if the target information was provided on the respective day or not:
	- Since the targets would only be available up to a certain time, he used a [[is present bit]] to represent if the targets exist or not
		- he did this because for up to 32 days of the competition, there won't be data present
		- ![[Pasted image 20240126112915.png]]
	- Non categorical features were scaled to roughly the same order
	- missing values were replaced by -1
	- [[create enriching features first, then mix across time]]
	- [[Adam optimizer]] with [[decreasing learning rate]] start: 2e-3 end: 5e-4
	- tried LSTM, CNNs or a transformer, but the GRU layer performed the best
	- normalization methods (batch norm, layer norm,…) didn't work
- (2nd)
	- https://www.kaggle.com/competitions/mlb-player-digital-engagement-forecasting/discussion/274661
	- cross validation: "We validated on the last month of data"
	- 
##### Important notebooks/discussions
- most popular eda
	- https://www.kaggle.com/code/chumajin/eda-of-mlb-for-starter-english-ver
#### Takeaways