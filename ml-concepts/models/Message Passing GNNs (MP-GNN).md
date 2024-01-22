- https://wandb.ai/graph-neural-networks/spatial/reports/An-Introduction-to-Message-Passing-Graph-Neural-Networks--VmlldzoyMDI2NTg2
	- they are graph neural networks but you are aggregating the features from nearby nodes. that's it!
- How are MP-GNNs different from [[Graph Convolutional Network (GCN)]]
	- People refer to them as the same things.
		- [This article](https://towardsdatascience.com/the-intuition-behind-graph-convolutions-and-message-passing-6dcd0ebf0063) does say that the difference is: message passing is a way to approximate graph convolutions.
			- Message passing is like taking into account the one-hop embeddings of your neighbouring nodes.
				- This is what one layer does in a GNN
			- if we stack n layers together, we're doing message passing n times
				- this is approximating a graph convolution that takes in features from n hops away
				- an analogy: in image convolutions, a 3x3 kernel takes in features from yourself and your neighbours. but we can also use a 5x5 kernel, which takes in even more features from further
				- stacking more layers is like increasing your kernel size

- main problems:
	- Since MPNNs are limited by problems of over-smoothing, over-squashing, and low expressivity against the WL test [1, 54], these layers could irreparably fail to keep some information in the early stage
		- https://openreview.net/pdf?id=lMMaNf6oxKM
	- 