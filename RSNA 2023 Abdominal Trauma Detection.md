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
	- smaller version of it is here: https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427427
- you 

##### Solutions
- (1st)
	- https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447449
		- solution code:
			- inference: https://www.kaggle.com/nischaydnk/rsna-super-mega-lb-ensemble
			- preprocessing and training: https://github.com/Nischaydnk/RSNA-2023-1st-place-solution
			- demo inference notebook: https://www.kaggle.com/code/haqishen/rsna-2023-1st-place-best-model-infer-cleaned
			- 3d segmentation: https://www.kaggle.com/code/haqishen/rsna-2023-1st-place-solution-train-3d-seg/notebook
		- stage 1:
			- image segmentation:
				- this identifies where organs are.
				- Then they crop these images based on 3D bounding box for the organ (found via img segmentation)
		- before stage 2
			- now that they have the bounding box for the organ (in 3D), they selected 96 equidistant slices
			- then they split the slices into three channels: (there are 32 x 3 slices)
			- why? cause (2, 32, 3, 384, 384) is much less compute than (2, 96, 1, 384, 384), shape
				- this is the shape stage 2 receives
			- [[resize layer to reduce dimensions]]
		- stage 2
			- they try to predict the masks again [[alternative targets (auxiliary objective)]]
		- how they prepared the target for stage 2
			- For example if a patient has label 0 for liver-injury and the liver visibility is as follows in the slice sequence
			- [0., 0., 0., 0.01, 0.05, 0.1, 0.23, 0.5, 0.7, 0.95, 0.99, 1., 0.95, 0.8, 0.4 …. 0. ,0., 0.]
			- We multiply it with label which is currently 0 results in an all zeros list as output, but if target label for liver-injury was 1, then we use the list mentioned above as our soft labels.
		- [[BCELoss]] for Classification, [[DiceLoss]] for segmentation
		- used [[image augmentation]]:
			- A.Perspective(p=0.5),
			- A.HorizontalFlip(p=0.5),
			- A.VerticalFlip(p=0.5),
			- A.Rotate(p=0.5, limit=(-25, 25)),
- (2nd)
	- https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447453
	- solution code:
		- Inference: https://www.kaggle.com/code/theoviel/rsna-abdominal-inf
		- training: https://github.com/TheoViel/kaggle_rsna_abdominal_trauma
	- [[GRU head (neck) after the backbone layer]]
		- "1. RNN only sees probabilities precomputed by the CNN, so training is done in 2 stages."
	- didn't do anything to address the data imbalance. the models handled it well
- (3rd)
	- https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447464
		- Crop: Since information around the organs is essential for trauma detection, the mask was slightly enlarged before the boxes were cut out. Two patterns of mask sizes were employed and two datasets were created for each organ.
		- The following are those that have made a particularly significant contribution to accuracy:  
			- masking for liver model  
			- custom sampler for all class models
				- "Since each organ has a different size, the appropriate image size and number of images should be different. Therefore, I used a custom sampler to ensure that only boxes of the same organ are included in the same batch. (If batch_size is 4, for example, 4 liver boxes will be included in one batch.)"
			- 2types of crops

##### Important notebooks/discussions

#### Takeaways
- It's hard to build an end-to-end model that predicts the target classes.
	- so ppl built models to identify organs via image segmentation, then derive the target classes from that
	- tbh. large models + more data will prob eliminate this intermediate step