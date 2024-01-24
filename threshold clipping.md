https://www.kaggle.com/competitions/google-quest-challenge/discussion/129978
- So let’s assume our predictions look like `x=[0.01, 0.015, 0.02, 0.03, 0.04]` and our optimal thresholds are `coefs=[0.016, 0.029]` then we would clip the data with `np.clip(x, coefs[0], coefs[1])` leading to `x=[0.016, 0.016, 0.02, 0.029, 0.029]` effectively generating ties at the edges.
- But how to generate these thresholds?
	- sample 1000 times single question-answer pairs from multiple questions
	- Generate thresholds that optimize the median score of all 1000 samples