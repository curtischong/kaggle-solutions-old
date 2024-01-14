Link: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states
Problem Type: 
Input: 
Output: 
Eval Metric: 
##### Summary
Detect sleep onset and wake from wrist-worn accelerometer data

##### Solution Links
(1st)
- https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459715
	- was based on this public notebook:
		- https://www.kaggle.com/code/danielphalen/cmss-grunet-train
			- uses [[KLDivergenceLoss]] because
				- 1) The target is "easily interpretable as a probability distribution"
				- 2) Ensembles are also easy to interpret, no matter how differently they are trained.
			- how to improve this notebook:
				- 1) use fastparquet
				- 2) add skip connections
				- 
	

#### Takeaways