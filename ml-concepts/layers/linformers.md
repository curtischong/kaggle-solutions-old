https://arxiv.org/pdf/2006.04768.pdf
- The main efficiency bottleneck in Transformer models is its self-attention mechanism. Here, each token’s representation is updated by attending to all other tokens in the previous layer
	- this is O(n^2)

https://www.youtube.com/watch?v=-_2AF9Lhweo
- if the matrix is low rank, we can speed it up
	- how to tell if a matrix is low rank? by looking at the eigenvalues of the matrix
	- a low rank matrix will have only a few eigen values (cause there are only a few significant vectors that have more info about the matrix)