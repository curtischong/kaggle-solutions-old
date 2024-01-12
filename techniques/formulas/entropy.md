Use this formula to determine how "sure" a model is of the output. If the entropy is high, then the model is probably predicting an unknown class

$$H = - \sum_i{p_i * \log(p_i)}$$

- where each pi is the probability of outcome i (aka class prediction) happening
- the negative sign is because the logarithm of a number between 0 and 1 is negative

#### what makes a higher entropy
1) Uniform distribution
	- cause if you're equally sure about all outcomes, you're not sure of one
2) Large Number of possible Outcomes
	- even if they don't have near probability
	- cause the more potential outcomes there are, the more unpredictable the event is.