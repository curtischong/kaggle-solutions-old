link: https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
range: [0, 1], (Real number)
### summary

- For two sets of items A (predicted Y) and B (target Y):
$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$

- For sentences:
	- given two sentences, split each sentence into tokens/words
	- find the number of tokens:
		- 1) that are the same
		- 2) that are different
	- calculate the jaccard distance

### pros

### Cons
