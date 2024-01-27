**Link:** https://www.kaggle.com/c/rsna-2023-abdominal-trauma-detection/overview
**Problem Type:** 
**Input:** 3D CT scans
**Output:** a probability for each of the different possible injury types:
- bowel_healthy,bowel_injury,extravasation_healthy,extravasation_injury,kidney_healthy,kidney_low,kidney_high,liver_healthy,liver_low,liver_high,spleen_healthy,spleen_low,spleen_high

**Eval Metric:** 
average of the sample weighted: [[log loss]]
- https://www.kaggle.com/code/metric/rsna-trauma-metric/notebook
##### Summary
- this is one of the biggest datasets I've seen. almost half a Terabyte of data
- you 

##### Solutions
- (1st)
	- https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447449
		- how they prepared the target for stage 2
			- For example if a patient has label 0 for liver-injury and the liver visibility is as follows in the slice sequence
			- [0., 0., 0., 0.01, 0.05, 0.1, 0.23, 0.5, 0.7, 0.95, 0.99, 1., 0.95, 0.8, 0.4 …. 0. ,0., 0.]
			- We multiply it with label which is currently 0 results in an all zeros list as output, but if target label for liver-injury was 1, then we use the list mentioned above as our soft labels.
		- [[BCELoss]] for Classification, [[DiceLoss]] for segmentation
- (2nd)
	- https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447453
	- solution code:
		- Inference: https://www.kaggle.com/code/theoviel/rsna-abdominal-inf
		- training: https://github.com/TheoViel/kaggle_rsna_abdominal_trauma
	- [[GRU head after the backbone layer]]
		- "1. RNN only sees probabilities precomputed by the CNN, so training is done in 2 stages."
	- didn't do anything to address the data imbalance. the models handled it well
- (3rd)

##### Important notebooks/discussions

#### Takeaways