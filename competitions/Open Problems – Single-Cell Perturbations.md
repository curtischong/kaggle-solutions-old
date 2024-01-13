Link: https://www.kaggle.com/competitions/open-problems-single-cell-perturbations
Problem Type: 
Input: cell types and small molecule names
Output: 
Eval Metric: [[Mean Rowwise Root Mean Squared Error]]
##### Summary
##### Solution Links
- (1st)
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

#### Takeaways