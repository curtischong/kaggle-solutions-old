- Sometimes we need to change our target labels to make it work better with our loss function
	- for example, we may use a custom form of [[label smoothing]]
- How to improve your labels
	- if your labels has the form: $\sum \frac{t}{r}$, consider using $\sum \frac{t^2}{r^2}$
		- play with how this looks:
			- https://www.desmos.com/calculator/gdehmkoyeq
		- the distribution of the labels looks like an inverse bell graph
			- there are more values with smaller labels, and more with larger labels