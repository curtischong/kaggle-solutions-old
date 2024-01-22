- GNN with learnable structural and positional representations: https://arxiv.org/pdf/2110.07875.pdf
	- For GNNs, the position of nodes is more challenging due to the fact that there does not exist a canonical positioning of nodes in arbitrary graphs
		- we can't just assign each node an index representing it's encoding since our model wouldn't be general to unseen graph topologies
		- a similar problem arises when we try to train the graph on Laplacian Eigenvectors
	- "In this work, we decouple structural and positional representations to make it easy for the network to learn these two critical characteristics. This is in contrast with most existing architectures s.a. Dwivedi & Bresson (2021); Beani et al. (2021); Kreuzer et al. (2021) that inject the positional information into the input layer of the GNNs"
		- and You et al. (2019) that rely on distance-measured anchor sets of nodes limiting general, inductive usage
	- they said that they were using "learnable positional encodings"
		- this is kinda sus, cause I'm not sure how they made it general for a graph with any toplogy
	- In this work, we consider two PEs: Laplacian PE (LapPE) and Random Walk PE (RWPE).
		- Laplacian PEs have this property Graph Positional encoding:
			- is permutation-invariant
			- is distance-sensitive
				- meaning that the difference between the PEs of two nodes far apart on the graph must be large, and small for two nodes nearby.
			- these are the spectral techniques that embed graphs into an Euclidean space
				- they are defined via the factorization of the graph Laplacian ∆ = In − D−1/2AD−1/2 = U TΛU, where In is the n × n identity matrix
			- Laplacian eigenvectors form a meaningful local coordinate system, while preserving the global graph structure. As these eigenvectors hold the key properties of permutation-invariant, uniqueness, computational efficiency and distance-aware
				- it seems like these eigenvectors are appended to nodes as features
			- 