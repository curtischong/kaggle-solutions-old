**Link:** https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting
**Problem Type:** [[Time Series]]
**Input:** game statistics for the current day
**Output:** a prediction of how engaged fans are with a specific player after a game
- there are 4 target columns, each representing the engagement level fans have with that player
- we are predicting the engagement levels 1 day after the game (i.e. 1 day after the x_train data)
**Eval Metric:** [[mean column-wise mean absolute error (MCMAE)]]
##### Summary

- Note: 
	- Binary columns will have null values as well as zeroes. Zeroes will occur if a player had an opportunity to do something, but did not. Nulls will occur if a player never had the opportunity to do something
		- e.g. a player who does not pitch on a given day cannot possibly pitch a shutout
		- how did they solve this?
##### Solutions

##### Important notebooks/discussions
- most popular eda
	- https://www.kaggle.com/code/chumajin/eda-of-mlb-for-starter-english-ver
#### Takeaways