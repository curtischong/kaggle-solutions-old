TODO. I am kinda confused. definitely needs more time to look. prob public kernels will help as well
Link: https://www.kaggle.com/c/cafa-5-protein-function-prediction/leaderboard
Problem Type: [[Multi-label Classification]]
Input: 
**Output:** for each protein, You have to predict the function (aka **GO term ID**) of a set of proteins based on their amino acid sequences and other data. There are thousands of possible classes (i.e. GO term IDs). Note: the classes are hierarchical: A leaf class can ONLY exist if all of its parent classes exist.

Eval Metric: [[F-score]]
- but the precision and recall (to calculate the F1 score) are weighted via the formulas on page 31 of this paper https://ndownloader.figstatic.com/files/7128245

The maximum F-measure based on the weighted precision and recall will be calculated on each of the three test sets and the final performance measure will be an arithmetic mean of the three maximum F-measures (for MF, BP, and CC). The formulas for computing weighted F-measures are provided in the [supplement](https://ndownloader.figstatic.com/files/7128245) (page 31) of the following [paper](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-1037-6)

Terms deep in the ontology tend to appear less frequently, be harder to predict, and thus their weights are larger (Clark & Radivojac, 2013). This does not always hold true however, as highlighted in the [following discussion](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/discussion/405237).

- TODO: understand why their metric implementation was flawed. (there was a discussion post). This will teach you how this metric actually works
##### Summary
The goal is to figure out "what each protein does" (it's function)


Note: This is like an oldschool kaggle competition. You submit predictions for the test directly via a tsv (tab separated value) file.
- so competitors know what the test targets are
##### Solutions

(2nd)
- https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/discussion/434064
- solution code: https://github.com/btbpanda/CAFA5-protein-function-prediction-2nd-place
	- they only did a simple 5-fold CV (nothing else was better)
	- [[Gradient-Boosted Decision Tree]] improved their models the most (their own pyboost framework is super fast)
(4th)
- https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/discussion/433732
	- approach
		- 1) Prot-T5, ESM2, and Ankh Protein Language Model (PLM) embeddings. We carried out no further modifications or finetuning on the output of PLMs, only conversion to float32 to save memory.
		- 2) A single binary matrix representing species taxonomy for each protein.
		- 3) used text information obtained by tf-idf of abstract information from academic papers associated with each protein
	- from comments:
		- concatenating ProtBERT sucked
		- [[dimension reduction for feature generation]] However, when I reduced the dimensions of ProtBERT to 3dims using [[UMAP dimension reduction]]/tSNE and added it, the score improved
	- 

(5th)
##### Important notebooks
getting started and understanding the competition: https://www.kaggle.com/code/gusthema/cafa-5-protein-function-with-tensorflow
- 
#### Takeaways


