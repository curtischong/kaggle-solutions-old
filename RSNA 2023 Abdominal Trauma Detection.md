**Link:** https://www.kaggle.com/c/rsna-2023-abdominal-trauma-detection/overview
**Problem Type:** 
**Input:** CT scans
**Output:** a probability for each of the different possible injury types:
- bowel_healthy,bowel_injury,extravasation_healthy,extravasation_injury,kidney_healthy,kidney_low,kidney_high,liver_healthy,liver_low,liver_high,spleen_healthy,spleen_low,spleen_high

**Eval Metric:** 
average of the sample weighted: [[log loss]]
- https://www.kaggle.com/code/metric/rsna-trauma-metric/notebook
##### Summary
- (1st)
	- https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447449
		- how they prepared the target for stage 2
			- For example if a patient has label 0 for liver-injury and the liver visibility is as follows in the slice sequence
			- [0., 0., 0., 0.01, 0.05, 0.1, 0.23, 0.5, 0.7, 0.95, 0.99, 1., 0.95, 0.8, 0.4 …. 0. ,0., 0.]
			- We multiply it with label which is currently 0 results in an all zeros list as output, but if target label for liver-injury was 1, then we use the list mentioned above as our soft labels.
		- - **Loss:** [[BCELoss]] for Classification, [[DiceLoss]] for segmentation
##### Solutions

##### Important notebooks/discussions

#### Takeaways