**Link:** https://www.kaggle.com/competitions/google-quest-challenge
**Problem Type:** [[learning to rank]]
**Input:** 
**Output:** 
**Eval Metric:** [[Spearman's correlation Coefficient]]
- The goal is to see **how well your model can rank** Q&A forum posts based on these target columns:
	- question_well_written
	- answer_helpful
	- answer_relevance
	- answer_type_reason_explanation
	- etc.
- there are 30 target columns
	- each target column is a value from [0,1]
	- for **each column**:
		- kaggle sorts your predictions (decreasing yhat)
			- the yhat you predict is just used to sort all the test rows
		- we now have an ordered array of [questionId, rank]
			- the rank is just the index of that questionId in the array
		- we now run spearman's correlation of this array AGAINST kaggle's rank of this array
	- since there are 30 columns, kaggle does it 30 times^
	- The final score is the average of all 30 of these spearman correlations
- A common approach is to treat it as a binary classification problem, cause the better you can predict each row's score, the more accurate it's rank will be
##### Summary

- okokoko There were A LOT OF THE TOP KAGGLE GRANDMASTERS IN THIS COMPETITION. just gushing sry
##### Solutions
(1st)
- https://www.kaggle.com/c/google-quest-challenge/discussion/129840
	- [[use intermediate layer results (weighted)]]
	- Using CLS outputs from all BERT layers rather than using only the last one. We did this by using a weighted sum of these outputs, where the weights were learnable and constrained to be positive and sum to 1
- some grandmasters are talking in the comments:
	- they didn't use thresholding, instead they placed each prediction into buckets (to generate the final result? not sure)
		- "But the threshold does a very similar thing. I actually tried both approaches on a very robust CV setup and fitting to distribution was always way worse for us. This was also confirmed when using rank-based approaches which inherenctly care more about the distribution."
		- "ANyway, binning and thresholding are quite similar"
- "Well, there is some post-processing. But tuning thresholds for each target was clearly seen as a straight way to overfitting, especially with 30 targets and so small public test size."
- "This model is initialized with original BERT weights, then finetuning with SX data, finally reusing the checkpoint in our pipeline. A subtle intermediate step was to delete classifier weights and biases from the SX checkpoint, because we had to switch from 6 SX targets to 30."
	- not sure if this is a trick or not. prob not
(2nd) - Very robust CV. Very nicely formatted target
- https://www.kaggle.com/competitions/google-quest-challenge/discussion/129978
	- Given the metric is rank-based, and given targets are not binary, it seemed important to be able to predict values that are neither 0 or 1 correctly.
		- they tried to use [[one-hot encoding]] of the targets (since some targets had a small number of distinct values)
			- I checked the dataset:
				- the "answer_type_procedure" column only had this as targets:
					- 0.000000, 0.333333, 0.666667, 1
		- 
	- ### Cross validation
		- [[GroupKFold]] didn't work
			- 

##### Important notebooks/discussions
- https://www.kaggle.com/code/carlolepelaars/understanding-the-metric-spearman-s-rho/notebook
	- many ppl were just using binary_crossentropy since the targets "look" like a binary classification problem

#### Takeaways