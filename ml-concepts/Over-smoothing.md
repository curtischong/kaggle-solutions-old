- GPT: different nodes in a graph neural network converge to similar representations, losing their distinctive features, as the amount of graph convolution or message passing layers increases
- GPT:
	- it happens cause:
		- **Iterative Aggregation**: each new layer does another round of message passing, which smooths features across nodes
		- **Homophily vs Heterophily**: 
			- similar nodes are more likely to be connected (homophily), in this case, oversmoothing can actually be beneficial
			- However, in graphs where dissimilar nodes are connected (heterophily), oversmoothing will lead to loss of these heterogenous aspects.
		- **Normalization Issue**:
			- normilization can amplify the oversmoothing problem by making the nodes' representations more similar after several convolutional layers
	- Strategies to combat it include early stopping, using models with residual or skip connections, and dynamic rational activation functions.
- https://arxiv.org/abs/2303.10993
	- just declares different ways to measure over smoothing
- https://www.reddit.com/r/MachineLearning/comments/tqdpf8/d_why_gnns_suffer_from_oversmoothing_but_cnns_dont/
	- inconclusive
- https://arxiv.org/abs/2305.16102
	- says: [[Graph Attention Networks (GATs)]] are susceptible to over-smoothing
		- basically: attention doesn't solve it
- https://medium.com/@xinyiwu98/oversmoothing-in-gnns-why-does-it-happen-so-fast-6bbe93ef97a7
	- Why does oversmoothing happen at a shallow depth?
		- Message-passing with different-class nodes homogenizes their representations exponentially.
		- Message-passing with nodes that have not been encountered before causes the denoising effect, and the magnitude depends on the absolute number of newly encountered neighbors.
		- The diameter of the graph is at most $\log N/\log(\log N)$ in our case.
		- After the number of layers surpasses the diameter, for each node, there will be no nodes that have not been encountered before in message-passing and hence the denoising effect will almost vanish. This is why even in a large graph, the mixing effect will quickly dominate the denoising effect when we increase the number of layers, and so oversmoothing is expected to happen at a shallow depth.