link: 
range: \[0, infinity)
### summary
- A mean absolute error is calculated for each of the n target variables and the score is the average of those n [[Mean absolute error (MAE)]] values.
- This is NOT the mean of all the targets
	- Here's a counterexample: 
		- we have two target columns: [1,3] and [99, 200, 500]
		- **MCMAE:**
		- $$\frac{1}{2}( \frac{1 + 3}{2} + \frac{99 + 200 + 500}{3}) = 134.166666667$$
		- **mean of all targets**
		- $$\frac{1 + 3 + 99 + 200 + 500}{5} = 160.6$$
### pros

### Cons