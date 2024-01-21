- Unlike GCNNs (e.g.  [[GraphSAGE]]), the Graph Attention Network (GAT) doesn't treat all neighbors equally. GAT introduces the attention mechanism that assigns different weights (or importance) to different neighbors
	- so rather than doing a simple sum of our neighbour's embeddings to calculate OUR next hidden embedding, **we do a weighted sum** of our neighbour's embeddings
		- the weight is determined by attention
- the main innovation: "A scoring function e : R d × R d →R computes a score for every edge (j, i), which indicates the importance of the features of the neighbor j to the node i"

GATv2: https://arxiv.org/abs/2105.14491
- cool implementation: https://nn.labml.ai/graphs/gatv2/index.html
- Attention is a mechanism for computing a distribution over a set of input key vectors, given an additional query vector. If the attention function always weighs one key at least as much as any other key, unconditioned on the query, we say that this attention function is static
	- this makes sense, cause the cool part about [[attention is all you need]] was how the query vector for each token is unique, and the magic was multiplying it with every token's key vector 
- TLDR: the original GAT paper didn't implement attention in the same spirit as the original [[attention is all you need]] paper. GATv2 fixes it.

- interesting video on graph attention
	- https://www.youtube.com/watch?v=CwsPoa7z2c8
	- this is the attention layer:
		- ![[Pasted image 20240121180335.png]]
	- basically, to get the "constant that tells us how much to weigh this neighbour's hidden layer", we just multiply the weight matrix with our hidden embedding (W * hi) concatted with the weight matrix with the neighbour's hidden embedding (W * hj)
