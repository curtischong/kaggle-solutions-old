good tutorial series: https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html
- solves these types of problems:
	- given this social graph, is it the same class as this other social graph?
	- given the classes of a few nodes in this graph, predict the class of nodes that don't have labels

- Why not just convert the graph into an adj matrix, and use image conv nets?
	- https://www.youtube.com/watch?v=CwsPoa7z2c8
		- here's how we'll construct the features as an adj matrix (for the conv net):
			- Each cell in the adj matrix are the feature vectors for edges
				- in reality, it'll be a 3D tensor, since each edge will prob have a feature vector.
			- We can also have another matrix for the feature vectors for nodes
				- e.g. row i of the matrix represents the features for node i
		- 2 problems:
			- 1) if the graph changes (new nodes/edges), we need to retrain the entire neural net
			- 2) If we change the labels of the graph, we'll get different adj matrices
				- ![[Pasted image 20240121175206.png]]
	- So we need to make a new kind of neural network to solve graph problems
[[Graph Convolutional Network (GCN)]]
	[[Spectral Graph Convolutions]]
[[Graph Attention Networks (GATs)]] (almost always outperforms GCNs)
[[GraphSAGE]]
[[graph isomorphism network]] (graph classification problem)
[[Relational Graph Convolutional Networks (R-GCNs)]]
[[Message Passing GNNs (MP-GNN)]]
[[Graph Segment Training]]
[[Graph Auto-Encoders (GAEs)]]
Variational Graph Auto-Encoders (VGAEs)
decent video explaining different schemes to incorporate edge features: https://www.youtube.com/watch?v=mdWQYYapvR8