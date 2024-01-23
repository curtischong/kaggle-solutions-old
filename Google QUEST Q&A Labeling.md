Sorry, I just wanted to add this competition because I read a really cool trick
**Link:** 
**Problem Type:** 
**Input:** 
**Output:** 
**Eval Metric:** [[Spearman's correlation Coefficient]]
##### Summary
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

##### Important notebooks/discussions

#### Takeaways