Link: https://www.kaggle.com/c/stanford-ribonanza-rna-folding
Problem Type: 
Input: 
Output: the reactivity for reactivity_DMS_MaP and reactivity_2A3_MaP for _each_ sequence position `id`
Eval Metric: [[MAELoss]]
##### Summary
##### Solutions

##### Important notebooks/discussions
- **How to check if your model generalizes to long sequences** https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/444653
	- x axis represents position, while y axis represents sequence number
	- Since the test set contains longer sequences than the train set, the model doesn't 'know' how to deal with unseen-before positions and does not generalize, i.e., the generated pictures are not similar to the ones posted by [@shujun717](https://www.kaggle.com/shujun717) above. As for other embeddings, I advise you to experiment and find out what works best
	- this picture is also a prediction, not ground truth. So it is possible that your model will do better than theirs.
		- so there's no point in trying to get your model to output a similar image
	- [[sliding window]]:
		- the idea: you train a transformer on sequences of 300 tokens.
			- How do you get predictions for sequences of 500 tokens?
			- Simple. You just use your 300-token transformer multiple times and take the average of the predictions.
				- you slide your 300-token transformer until all 500 tokens were in at least one prediction
		- Why they thought it was good for generalization:
			- cause the training data was at most 206 long, but the test data is at most 457 long
			- so to generalize their models (trained on the shorter training data), they could apply the sliding window on the longer test data examples.
		- However, it didn't work, since it is "wrong in a biological sense" (1st place writeup)
	- comments
		- About the sliding window, I'm not sure what is the benefit of using it with such short sequences, the main idea behind sliding window is to reduce the complexity from O(n^2) to O(n * w) where w is smaller than n but the RNA sequences of this competition are <= 457, so the normal attention can be computed without much trouble.
			- (1st place): "sliding window seems to be useless"
			- It can be useful in this problem, since in train set Lmax=206, and in test Lmax=457
		
				My guess was if I am to use sliding window attention, my generalization plots would become closer to the ones Shujun shared, but it was not the case, they are still "too sharp"
- 
#### Takeaways