**Link:** https://www.kaggle.com/c/icr-identify-age-related-conditions
**Problem Type:** [[Multi-label Classification]] (since ppl can have more than one condition) - however, the submission is [[binary classification]]
**Input:** 
**Output:** predict if a person has any of three medical conditions (class 1) or not (class 0)
- Note: the training data has `B`, `D`, `G` The three age-related conditions. Correspond to class `1`.
**Eval Metric:** 
##### Summary
- ![[Pasted image 20240128173618.png]]
	- yep :) This is a fun competition
- Note: All of the data in the test set was collected after the training set was collected.
##### Solutions
- (1st)
	- https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/430843
	- [[TabPFN]]
	- The "greeks.csv" was useless. I think, because we have no greeks for the test data.
	- Gradient boosting was obviously overfitting
	- feature engineering led to overfitting
	- What did work:
		- DNN based on Variable Selection Network
			- https://arxiv.org/abs/1912.09363
		- No "casual" normalization of data like MinMaxScaler or StandartScaler, but instead a linear projection with 8 neurons for each feature.
		- Huge values of dropout: 0.75->0.5->0.25 for 3 main layers.
		- Reweighting the probabilities in the end worked really good
		- 10 folds cv, repeat for each fold 10-30 times, select 2 best models for each fold based on cv (yes, cv somehow worked in this competition!).The training was so unstable, that the cv-scores could vary from 0.25 to 0.05 for single fold, partially due to large dropout values, partially due to little amount of train data. That's why I picked 2 best models for each fold.
		- The cv was some kind of Multi-label. At first I trained some baseline DNN, gathered all validation data and labeled it as follows: (y_true = 1 and y_pred < 0.2) or (y_true = 0 and y_pred > 0.8) -> label 1, otherwise label 0. So, this label was somthing like "hardness to predict". And the other label was, of course, the target itself.
	- target engineering
		- Added a "hardness to predict" label
				- 
	- feature engineering
		- imputed missing values with the median [[filling training data]] (didn't test mean / [[K-nearest neighbour (KNN)]])
		- didn't  resample or preprocess for noise reduction
	- the baseline model that came up with 'hard to predict' classification is:
		- "literally the model with 2 VariableSelectionFlow layers instead of 3"

##### Important notebooks/discussions
- Baseline notebook and EDA
	- https://www.kaggle.com/code/gusthema/identifying-age-related-conditions-w-tfdf
		-  positive samples (class 1) only account for 17.50% of the data
- Explaining why the problem is hard
	- https://www.kaggle.com/code/raddar/icr-competition-analysis-and-findings
		- The BN column is age [[distribution matching]]
		- The Population is segmented by the `EJ` column
			- when `EJ == 0`, `EH = 0.5`. This means there is no reason to encode `EJ` [[drop redundant columns]]
		- there are more class1 examples for the rows that occur on later years
			- ![[Pasted image 20240128144336.png]]
		- They ran a [[TSNE]] and [[K-nearest neighbour (KNN)]] on the data.
			- Found that for all the `class 0` examples, there are only 8 `class 1` examples that are near it
			- basically: after clustering, there are very few `class 1` examples that are intermixed with the `class 0` examples.
			- This means that there are very FEW "hard examples".
				- This is BAD because correctly deducing the correct class for hard examples is what makes the model stronger
- using the greeks.csv supplimental data
	- https://www.kaggle.com/code/sugataghosh/icr-the-devil-is-in-the-greeks
	- The key takeaway from this notebook is the observation that these supplemental metadata, in particular the categorical variables `Beta`, `Gamma` and `Delta`, are far more powerful in predicting `Class`, compared to the features provided in the `train` dataset.
	- **It may be helpful to first predict the categorical variables `Beta`, `Gamma` and `Delta` in the metadata for the test set, and then predict the `Class` based on the predicted metadata variables.**
#### Takeaways