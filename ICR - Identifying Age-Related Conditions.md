**Link:** https://www.kaggle.com/c/icr-identify-age-related-conditions
**Problem Type:** [[binary classification]]
**Input:** 
- `AB`-`GL` Fifty-six anonymized health characteristics. All are numeric except for `EJ`, which is categorical.
**Output:** predict if a person has any of three medical conditions (class 1) or not (class 0)
- Note: the Supplemental metadata has `B`, `D`, `G` The three age-related conditions. Correspond to class `1`.
	- NOTE: (1st) did not use this (since it isn't available for the test set)
**Eval Metric:** 


- The submitted probabilities for a given row are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum)
	- since we're predicting class 0 and class 1, and it's rebalanced, this is just a binary classification
##### Summary
- ![[Pasted image 20240128173618.png]]
	- yep :) This is a fun competition
	- **7,327 Competitors, 6,430 Teams**
- Note: All of the data in the test set was collected after the training set was collected.
- **Why did people think that CV didn't Work**
- **Why did people think their CV was good, but got bad scores?**
- **What is reweighing?**
##### Solutions
- (1st) [[TabPFN]] plus CV + [[alternative targets (auxiliary objective)]] for "hard to predict"
	- https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/430843
	- solution code: https://storage.googleapis.com/kaggle-forum-message-attachments/2384935/19562/adv_model_training.ipynb
	- The "greeks.csv" was useless. I think, because we have no greeks for the test data.
	- Gradient boosting was obviously overfitting
	- feature engineering led to overfitting
	- **Cross validation**
		- 10 folds cv, repeat for each fold 10-30 times, [[select 2 best models for each fold on CV]] (yes, cv somehow worked in this competition!).The training was so unstable, that **the cv-scores could vary from 0.25 to 0.05 for single fold**, partially due to large dropout values, partially due to little amount of train data. That's why I picked 2 best models for each fold.
		- in the training notebook:
			- their CV is using StratifiedKFold
			- they are only printing the lowest validation loss (not using? [[select 2 best models for each fold on CV]])?
				- the weights are saved to their own h5 file. so maybe they're just picking the right one based on the score
			- I can confirm that for each of the 10 train-validate splits, they are running them 10 times.
	- What did work:
		- DNN based on Variable Selection Network
			- https://arxiv.org/abs/1912.09363
		- No "casual" normalization of data like MinMaxScaler or StandartScaler, but instead a linear projection with 8 neurons for each feature.
		- Huge values of dropout: 0.75->0.5->0.25 for 3 main layers.
		- Reweighting the probabilities in the end worked really good
		- The cv was some kind of Multi-label. At first I trained some baseline DNN, gathered all validation data and labeled it as follows: (y_true = 1 and y_pred < 0.2) or (y_true = 0 and y_pred > 0.8) -> label 1, otherwise label 0. So, this label was somthing like "hardness to predict". And the other label was, of course, the target itself.
	- target engineering
		- Added a [[hardness to predict label]]
			- Trained some baseline DNN, gathered all validation data and labeled it as follows: (y_true = 1 and y_pred < 0.2) or (y_true = 0 and y_pred > 0.8) -> label 1, otherwise label 0
			- this DNN is: "literally the model with 2 VariableSelectionFlow layers instead of 3"
	- feature engineering
		- imputed missing values with the median [[filling training data (impute data)]] (didn't test mean / [[K-nearest neighbour (KNN)]])
		- didn't  resample or preprocess for noise reduction
	- comments
		- "How much did you step 6 improve your model. If I get it right, you override original targets by the model's out of fold prediction if the prediction is confident enough. I had something like that in mind but it seems it felt in a crack. I used this successfully in the past and I have seen recent competition winners share something similar too." - CPMP, Kaggle Grandmaster
			- "I didn't override the original target, the tgt2 was used only for better splits during cross-validation. Step 6 improved the score by 0.02."
		- "In my experiments standardization of the data deteriorated the cv-score, though I don't fully understand why"
		- "It is a bit funny to see people thinking Time series should behave like I.I.D and stationary data . These deviations are not uncommon , particularly in fields like Finance. That's why modelling for these data is still very challenging and overfitting them is very easy" - Serigne - competitions master
		- "I remember clearly conventional belief that we are not even suppose to try neural network if we don't have a sizeable data. I guess I'm dead wrong."
		- "I tried reducing dimensions with autoencoder, but it didn't look very promising." - Tilii - competitions master
		- it took 10 hrs to train the DNN
		- "And regarding [[balanced logloss]] - as I understand this, reweighting just stretches the probabilities in one or other side, but if they are in the middle (0.5) they stay there)"
- (2nd)
	- https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/430860
	- filled Nan values with -100 [[filling training data (impute data)]]
	- used a technique I saw in some other competition: reducing dimensions with umap and then labeling clusters with kmeans, it didn't bring a lot of score, but it's there
	- did feature permutation manually, dropped any cols that made score even slightly worse
		- interesting that it worked
	- Catboost, xgb with parameters from some public notebook and [[TabPFN]]
maybe " the reweighting is done in a way such that the class predictions are more balanced overall across all test data rows"

##### Important notebooks/discussions
- Baseline notebook and EDA
	- https://www.kaggle.com/code/gusthema/identifying-age-related-conditions-w-tfdf
		-  positive samples (class 1) only account for 17.50% of the data
- Explaining why the problem is hard (why CV is hard to make)
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
		- however, this method of finding "close" samples is fiddly. if you use diff features, clusters could be easier/harder to form
	- "The fact that the evaluation metric is almost entirely determined by the presence or absence of the hard cases (false negatives) means that any CV (or indeed LB) evaluation is not going to be a continuous function, let alone differentiable. This explains the difficulties in getting a robust CV scheme we've seen in this competition (as well as the crazy leaderboard)."
- using the greeks.csv supplimental data
	- https://www.kaggle.com/code/sugataghosh/icr-the-devil-is-in-the-greeks
	- The key takeaway from this notebook is the observation that these supplemental metadata, in particular the categorical variables `Beta`, `Gamma` and `Delta`, are far more powerful in predicting `Class`, compared to the features provided in the `train` dataset.
	- **It may be helpful to first predict the categorical variables `Beta`, `Gamma` and `Delta` in the metadata for the test set, and then predict the `Class` based on the predicted metadata variables.**
	- NOTE (1st) didn't use this supplimental data cause it's not in the test set
- introducing [[TabPFN]] and reweighing
	- https://www.kaggle.com/code/muelsamu/simple-tabpfn-approach-for-score-of-15-in-1-min
	- code
		```python
		    def predict_proba(self, X):
		        X = self.imp.transform(X)
		        ps = np.stack([cl.predict_proba(X) for cl in self.classifiers])
		        p = np.mean(ps,axis=0)
		        class_0_est_instances = p[:,0].sum()
		        others_est_instances = p[:,1:].sum()
		        # we reweight the probs, since the loss is also balanced like this
		        # our models out of the box optimize CE
		        # with these changes they optimize balanced CE
		        new_p = p * np.array([[1/(class_0_est_instances if i==0 else others_est_instances) for i in range(p.shape[1])]])
		        return new_p / np.sum(new_p,axis=1,keepdims=1)
		```
- Similarity CV (doesn't work): "A novel way to tackle the problem"
	- https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/414856
		- "instead of focusing on individual of the sample, we consider a couple of individuals (𝑖,𝑗) and try to guess if they belong to the same class"
		- Now our new training dataset has $N^2$ samples, the new validation dataset has $N * M_0$ samples
			- where $M_0$ is the size of our validation set
		- kaggle master: "I think it would work well if the two classes were better separated overall. Sadly, I don't think it will be competitive for this dataset"
#### Takeaways
- Use a deep neural network. Always. Even if you have little data. If you have a good CV, it'll work.
- (1st) "feature engineering led to overfitting". This requires blindly believing in your CV. which is so painful to accept