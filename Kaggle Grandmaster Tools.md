 Stochastic Weights Averaging -  Saves n the best epochs states and then average it in final model.
- https://github.com/btbpanda/CAFA5-protein-function-prediction-2nd-place/blob/main/protnn/swa.py#L8

Performance:
- from numba inport jit
	- @jit(nopython=True)
- https://github.com/nalepae/pandarallel
- using cupy as a near drop-in replacement for numpy (to do ops with GPU)
	- you need to wrangle with cp.cuda.MemoryPool()

[[Graph Convolutional Network (GCN)]]
- https://www.dgl.ai/