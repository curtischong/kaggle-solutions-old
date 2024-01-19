used in [[binary classification]] problems
```
L(y, f(x)) = max(0, 1 - y * yhat)
```
- Where y is either +1 or -1
- Notice how if y is the same sign as yhat, the `1 - y * yhat` term becomes smaller
	- if this term is less than 0, the max function just makes the loss 0!
- So the more items that you classify correctly, the smaller the loss.