https://www.youtube.com/watch?v=qA6U4nIK62E
- like autoencoders, their goal is to produce a matrix representation of the input
	- $\bar{X} = GCN(A, X) = ReLU(\bar{A}XW_0)$
		- $\bar{X}$ is the representation
	- Note that $\bar{A}$ is the normalized adj matrix:
		- $\bar{A} = D^{-1/2} AD^{-1/2}$
- ![[Pasted image 20240121201228.png]]
- how to reconstruct the input graph?
	- Solution 1:
		- Perform an inner product all indexes in the latent space (and sigmoid the result) to get the adj matrix back
			- ![[Pasted image 20240121201430.png]]
		- why does this work?
- variational autoencoders
	- 