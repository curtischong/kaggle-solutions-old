TODO. I am kinda confused. definitely needs more time to look. prob public kernels will help as well
Link: https://www.kaggle.com/c/cafa-5-protein-function-prediction/leaderboard
Problem Type: 
Input: 
Output: You have to predict all of the possible classes each gene can have. There are thousands of possible classes. Note: the classes are hierarchical: A leaf class can ONLY exist if all of its parent classes exist.
Eval Metric: 
The maximum F-measure based on the weighted precision and recall will be calculated on each of the three test sets and the final performance measure will be an arithmetic mean of the three maximum F-measures (for MF, BP, and CC). The formulas for computing weighted F-measures are provided in the [supplement](https://ndownloader.figstatic.com/files/7128245) (page 31) of the following [paper](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-1037-6)

Terms deep in the ontology tend to appear less frequently, be harder to predict, and thus their weights are larger (Clark & Radivojac, 2013). This does not always hold true however, as highlighted in the [following discussion](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/discussion/405237).

- TODO: understand why their metric implementation was flawed. (there was a discussion post). This will teach you how this metric actually works
##### Summary
The goal is to figure out "what each protein does" (it's function)
##### Solution Links

(2nd)
- https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/discussion/434064
	- they only did a simple 5-fold CV (nothing else was better)
	- [[Gradient-Boosted Decision Tree]] improved their models the most (their own pyboost framework is super fast)
- 
#### Takeaways