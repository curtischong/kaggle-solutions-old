Link: https://www.kaggle.com/competitions/open-problems-single-cell-perturbations
Problem Type: 
Input: cell types and small molecule names
Output: 
Eval Metric: [[Mean Rowwise Root Mean Squared Error]]
##### Summary

Note: an estimated 35% of the training data is erroneous: https://www.kaggle.com/code/jalilnourisa/post-eda
- This could affect the validity of techniques used in this competition
##### Solution Links
- (1st) ChemBERTa
	- https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/discussion/459258
	- generated features by embedding the description of the cell
		- but these text embeddings decreased his score.
		- He also tried fine-tuning these text embeddings but it didn't work
	- cross validation: 5fold CV
	- ChemBERTa embeddings of the SMILES encodings helped tremendously
	- other features he used: the mean, standard deviation, and (25%, 50%, 75%) percentiles per cell type and small molecule
		- TODO: what is he aggregating over? A cell type is a class, not a number!
	- feature representations
		- initial”: ChemBERTa embeddings, 1 hot encoding of cell_type/sm_name pairs, mean, std, percentiles of targets per cell_type and sm_name  
		- “light”: ChemBERTa embeddings, 1 hot encoding of cell_type/sm_name pairs, mean targets per cell_type and sm_name  
		- “heavy”: ChemBERTa embeddings, 1 hot encoding of cell_type/sm_name pairs, mean, 25%, 50%, 75% percentiles of targets per cell_type and sm_name
	- model selection:
		- didn't work: gradient boosting models, MLP, and 2D CNN
		- worked: LSTM, GRU, 1D CNN
	- Loss function:
		- 0.32[[MSELoss]] + 0.24[[MAELoss]] + 0.24[[LogCosh]] + 0.2[[BCELoss]]
			- Although BCE is for binary classification, it was used cause it "sends better signals to the models and optimizers when the target values are close to zero"
			- e.g.
				- BCELoss(0.05, -0.05) = 0.694
				- MSELoss(0.05, -0.05) = 0.010 # this loss is much smaller! However, the values are quite far!
			- Since most target values are from a Gaussian distribution with mean 0, using BCELoss will more aggressively punish imprecise predictions
	- removed padding in the ChemBERTa model improved private leaderboard results
	- also setting 30% of the input features’ entries to 0 improves the score
		- the hypothesis is:
			- 1) we might not need to know the complete chemical structure of a molecule to know its impact on a cell. OR
			- 2) there is a biological disorder in the cell, but we still expect it to respond to the drug in the same way
		- this feels like [[dropout]]
		- NOTE: significant training data qualities could mean this technique isn't valid
- (2nd) 
	- https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/discussion/458738
	- used one-hot encoding
	- [[cluster sampling]]
	- To address high and low bias labels, I utilized target encoding by calculating the mean and standard deviation for each cell type and SM name
#### Takeaways