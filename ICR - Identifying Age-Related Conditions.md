**Link:** https://www.kaggle.com/c/icr-identify-age-related-conditions
**Problem Type:** [[Multi-label Classification]] (since ppl can have more than one condition) - however, the submission is [[binary classification]]
**Input:** 
**Output:** predict if a person has any of three medical conditions (class 1) or not (class 0)
- Note: the training data has `B`, `D`, `G` The three age-related conditions. Correspond to class `1`.
**Eval Metric:** 
##### Summary
- Note: All of the data in the test set was collected after the training set was collected.
##### Solutions

##### Important notebooks/discussions
- Baseline notebook and EDA
	- https://www.kaggle.com/code/gusthema/identifying-age-related-conditions-w-tfdf
		-  positive samples (class 1) only account for 17.50% of the data
- Explaining why the problem is hard
	- https://www.kaggle.com/code/raddar/icr-competition-analysis-and-findings
		- The BN column is age [[distribution matching]]
		- The Population is segmented by the `EJ` column
			- when `EJ == 0`, `EH = 0.5`. This means there is no reason to encode `EJ` [[drop redundant columns]]

#### Takeaways