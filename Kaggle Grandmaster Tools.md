 [[Stochastic Weights Averaging]] -  Saves the best n epochs states and then averages them in final model.
- https://github.com/btbpanda/CAFA5-protein-function-prediction-2nd-place/blob/main/protnn/swa.py#L8

Performance:
- from numba inport jit
	- @jit(nopython=True)
- https://github.com/nalepae/pandarallel
- using cupy as a near drop-in replacement for numpy (to do ops with GPU)
	- you need to wrangle with cp.cuda.MemoryPool()

[[Graph Convolutional Network (GCN)]]
- https://www.dgl.ai/

Project / model management:
- https://www.kaggle.com/c/google-quest-challenge/discussion/129840
	- " About 2 years ago I spent a couple of evenings to create a very tiny python library called [mag](https://github.com/ex4sperans/mag) that addressed the problem of machine learning experiment management. Since then, I used it in every single competition including this one. This really helped our team to don't lose track of all the models we trained for these two weeks. I hope some of you might find it useful too."
	- https://github.com/ex4sperans/maggot
- ppl on kaggle aren't talking about DVC
	- prob cause they keep on changing their feature engineering, and it's a static dataset
	- I think it would be very useful for hyperparm tuning, but it's prob not a bottleneck for most ppl