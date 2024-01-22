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
- (1st) Speed up iteration time. Custom attention layer to compare configs throughout the network
	-  https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/456343
		- solution code: https://github.com/thanhhau097/google_fast_or_slow/tree/main
		- We pruned and compressed the layout graphs in order to increase the efficiency of our experiments [[speedup iteration]]
			- They noticed that only `Convolution`, `Dot` and `Reshape` were configurable nodes
			- So their pruning strat was: for each graph, only keep the nodes that were either configurable models themselves or were connected to a configurable node
			- this reduced vram usage by 4x and increased training speed by 5x
		- deduplication: they found that configs for different network layouts had many duplications
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
		- network architecture
			-  for the graph convolutional layer itself:
				- [[SAGEConv]] was good
					- GAT variants didn't work (Graph attention)
				- but self-channel attention and cross-config attention was good
			- [[Squeeze-and-Excitation layer]]
			- crossConfigAttention
				- Another dimension that we can exploit attention is the batch plane (cross-configs). We designed a very simple block that allows the model to explicitly "compare" each config against the others throughout the network. We found this to be much better than letting the model infer for each config individually and only compare them implicitly via the loss function (`PairwiseHingeLoss`). The attention code is as follows:
					- code
						```python
						class CrossConfigAttention(nn.Module):
						    def __init__(self):
						        super().__init__()
						        self.temperature = nn.Parameter(torch.tensor(0.5))
						
						    def forward(self, x):
						        # x of shape (nb_configs, nb_nodes, nb_features)
						        scores = (x / self.temperature).softmax(dim=0)
						        x = x * scores
						        return x
						```
	
				- By applying this simple layer after the self-channel attention at every block of the network, it gave us a huge boost for default collections.
				- it's applied like so:
					- ![[Pasted image 20240119021535.png]]
				- Note: the output of the forward layer is (nb_configs, nb_nodes, nb_features) since multiplication and sigmoid are scalar operations
					- cause dim=0 means: This operation will get probabilities that sum up to 1 across the dimension 0
				- I'm not sure why they concat the cross attention with the self channel attention. I don't think they need to do that?
					- oh it's answered: (we concat the output with its input to preserve the individuality of each sample)
				- note that they do: `scores = (x / self.temperature)`
					- but I think that scores = (x * self.temperature) works as well since self.temperature is used once.
		- However, since the prediction depends on the batch, we can leverage it further by applying TTA (Test Time Augmentation) to generate `N` (10) permutations of the configs and average the result after sorting it back to the original order.
			- I think the author is saying: inferencing is unstable. So to get better answers, we generate 10 permutations of the same config, then average the result, so the inferred value is smoohter.
			- but I don't get it. ppl usually say this when ensembling, not "in a layer". also why does he mention batch size during inference?
			- https://github.com/thanhhau097/google_fast_or_slow/blob/626c463dfb02abd739616773ca74f34e38635c71/dataset.py#L441
				- I think he just means that he's enssembling the results 10 times
				- he does: # shuffle runtime and config feats
					- https://github.com/thanhhau097/google_fast_or_slow/blob/626c463dfb02abd739616773ca74f34e38635c71/dataset.py#L490
		- To create our Linear/Conv blocks we followed the good practices in computer vision. We start by using `InstanceNorm` to normalise the input feature map, followed by `Linear`/[[SAGEConv]] layer, `SelfChannelAttetion` and `CrossConfigAttetion` (we concat the output with its input to preserve the individuality of each sample). Then, we sum the residual connection and finish with [[Gaussian Error Linear Unit (GELU)]] and dropout.
		[[PairwiseHingeLoss]]
- (2nd) Invented their own DiffMat Loss
	- https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/456365
	- solution code: https://github.com/Obs01ete/kaggle_latenciaga/tree/master
	- data cleaning
		- like (1st), they found duplicated configs (with diff runtimes). so they deduped them, and keept the config that had the lowest runtime
		- didn't use unet graphs cause `unet_3d.4x4.bf16` was badly corrupted
		- they repackaged each NPZ so that each config+runtime measurement could be loaded without needing to load the entire NPZ
	- model
		- All GNN layers are SageConv layers with residual connections whenever the number of input and output channels are the same.
		- Features produced by the GNN layer stack are transformed to one value per node and then sum-reduced to form a single graph-wise prediction.
			- not sure what sum-reduced means. but probably mean?
	- training
		- Models trained with a ranking loss ([[ListMLE Loss]], [[marginRankingLoss]]) heavily outperformed element-wise losses (MAPE, etc).
		- "The batch is organized into 2 levels of hierarchy: the upper level is different graphs, and the lower level is the same graph and different configurations, grouped in microbatches of the same size (also known as slates)"
			- todo: understand
		- hyperparams
			1. Adam/AdamW optimizer,
			2. Learning rate 1e-3,
			3. 400k iterations,
			4. Step learning rate scheduler at 240k, 280k, 320k, and 360k by factor of `1/sqrt(10)`.
		- Losses used for training:
			1. [[ListMLE Loss]] for Layout-NLP,
			2. A novel DiffMat loss for Tile,
			3. For Layout-XLA, it is a combination of 2 losses: the DiffMat loss and MAPE loss.
	- DiffMat Loss
		- The idea is to use the difference matrix between ...
			- the difference matrix reminds me of [[marginRankingLoss]] (since it's euclidian distance)
		- this is fed into [[marginRankingLoss]]
		- I'm not sure what "Margin Ranking Loss with a margin of 0.01 is applied between predicted values and zeros" means
			- this diagram is kinda vague:
				- ![[Pasted image 20240120015431.png]]
			- I tried reading their code but couldn't find it.
	- data recovery
		- We tried to find the damaged data and remove it in an automatic manner by computing block-wise [[entropy]] of the runtimes between adjacent blocks. 
	- what didn't work:
		- [[Online hard negative mining]] "since train loss is nowhere near zero,"
			- I think this is because the model already isn't learning positive OR negative examples well
			- The model needs to first learn well from the "easier" examples before introducing "harder" ones.
		- Train 4 folds and merge by mean latency and by [[mean reciprocal rank (MRR)]]
- (3rd) Novel feature generation
	- https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/456377
	- [[linformers]]
	- Used [[GNN Positional encodings]]
	- [[GPS Layers]]
	- [[adam optimizer]], [[cosine annealing LR]]
- (4th) Only uses a multilayer perceptron
	- https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/456462
- (5th)
	- Tips
		- use the same opcode embedding for unary operations such as abs, ceil, cosine, etc.
		- override layout_minor_to_major by layout config features for configurable nodes
		- [[DropEdge]]
		- apply log transformation to input features
		- oversampling
		- load layout config data by numpy's mmap mode to save RAM
	- What Didn't Work
		- graph pooling
		- pretrain on the tile dataset and finetune on the layout dataset
		- graph normalization
		- dropout node
		- GAT, GATv2, GIN
			- GATV2 DID NOT WORK???!?!??!
		- fp16
		- pseudo label
##### Important notebooks/discussions
- understanding the competition
	- https://www.kaggle.com/code/ayushs9020/understanding-the-competition-google-slow-vs-fast
- explaining tensor tile, tensor shard, simulated annealing, and langevin dynamics
	- https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/435577
- video links to googlers explaining mroe about the problem, and considerations
	- https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/436629

#### Takeaways