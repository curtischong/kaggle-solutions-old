**Link:** https://www.kaggle.com/c/LANL-Earthquake-Prediction
**Problem Type:** 
**Input:** 
**Output:** 
**Eval Metric:** [[Mean absolute error (MAE)]]
##### Summary
- you’ll predict the time remaining before laboratory earthquakes occur from real-time seismic data.
- these aren't real earthquakes. it's a lab setup:
	- ![[Pasted image 20240129154843.png|400]]
- **Why did many ppl overfit the public lb?**
	- 
##### Solutions
- (1st) PSI's solution
	- "After doing abovementioned signal manipulation, we had more trust in our calculated features and could focus on better studying differences between train and test data feature distributions"
	- we calculated a handful of features for train and test and tried to find a good subset of full earth-quakes in train, so that the overall feature distributions are similar to those of the full test data.
	- [[KS statistic]]
	- a simple shuffled 3-[[kfold]]
	- [[adversarial validation]]
##### Important notebooks/discussions

#### Takeaways