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
	- solution code: https://github.com/thanhhau097/google_fast_or_slow/tree/main
	- We pruned and compressed the layout graphs in order to increase the efficiency of our experiments [[speedup iteration]]
		- They noticed that only `Convolution`, `Dot` and `Reshape` were configurable nodes
		- So their pruning strat was: for each graph, only keep the nodes that were either configurable models themselves or were connected to a configurable node
		- this reduced vram usage by 4x and increased training speed by 5x
	- deduplication: they found that configs for different network layouts had many dupications
		- and "the runtime for the duplicated configs can vary quite a bit and make training less stable"
		- why would the runtime be different if they were the same?
			- probably because of page faults? or leaky memory in their computer?
				- or maybe on subsequent runs, they were cached
			- I guess the first run would be the most accurate (since it's fresh)
	- loading all the data still used lots of ram
	- so they compressed node_config_feat and only decompressed it on-the-fly in the dataloader [[data compression]]
		- they realized that each of the 6 columns in node_config_feat only has 7 possible values
			- so each row can be represented as a base 7 integer: from 0 to 7^6
		- they stored the compressed data as a numpy array
			- https://github.com/thanhhau097/google_fast_or_slow/blob/main/data/data_compression.py
			- code
				```python
				def vec_to_int(vec: np.ndarray) -> np.ndarray:
				    # Powers of 7: [1, 7, 49, 343, 2401, 16807]
				    powers_of_7 = np.array([7**i for i in range(6)])
				    return np.dot(vec, powers_of_7).astype(np.int32)
				```
	- by loading all data to memory at the beginning of training, they reduced IO/CPU bottlenecks considerably and allowed them to train faster. [[speedup iteration]]
	- **change pad value in `node_feat` from 0 to -1**
		- they realized that values in `node_config_feat` was -1 padded, but values in `node_feat` was 0-padded
			- this was a problem cause for some features 0 is a valid value
		- so they changed `node_feat` to be -1 padded so that they can [[use a single embedding matrix]] for both `node_feat[134:]` and `node_config_feat` [[placeholder for invalid values]]
	- [[normalize features]]
		- They used `StandardScaler` for these features: `node_feat[:134]`
			- cause some features were `*_sum` and `*_product`, which can have high values
			- these disrupt the optimization
	[[PairwiseHingeLoss]]
	

##### Important notebooks/discussions
- understanding the competition
	- https://www.kaggle.com/code/ayushs9020/understanding-the-competition-google-slow-vs-fast
- explaining tensor tile, tensor shard, simulated annealing, and langevin dynamics
	- https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/435577
- video links to googlers explaining mroe about the problem, and considerations
	- https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/436629

#### Takeaways