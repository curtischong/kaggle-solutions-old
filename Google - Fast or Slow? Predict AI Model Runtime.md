**Link:** https://www.kaggle.com/c/predict-ai-model-runtime
**Problem Type:** 
**Input:** 
**Output:** 
**Eval Metric:** [[Kendall Tau correlation]]
##### Summary
the goal is to predict the runtime of an AI model based on its characteristics, such as the number of parameters/the number of layers/hardware configuration.
- you have to predict the runtime for many different types of ml models
	- e.g. bert, resnet50

glossary:
- xla means: accelerated linear algebra. it's an open-source compiler for machine learning: https://www.tensorflow.org/xla
- Hlo means: high level optimizer
##### Solutions
(1st)
-  https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/456343
	- We pruned and compressed the layout graphs in order to increase the efficiency of our experiments [[speedup iteration]]
[[PairwiseHingeLoss]]

##### Important notebooks/discussions
- understanding the competition
	- https://www.kaggle.com/code/ayushs9020/understanding-the-competition-google-slow-vs-fast
- explaining tensor tile, tensor shard, simulated annealing, and langevin dynamics
	- https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/435577
- video links to googlers explaining mroe about the problem, and considerations
	- https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/436629

#### Takeaways