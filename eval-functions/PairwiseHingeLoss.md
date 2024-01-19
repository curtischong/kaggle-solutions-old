- used in ranking problems where we are predicting a **positive instance** and **negative instance**.
	- basically, it means that the model outputs two scores, the positive instance is > 0 the negative instance is < 0
- an extension of [[HingeLoss]]
- the loss:
	```
	L(max(0, margin - [score_u - score_v]))
	```
	- where margin = 1 (in most cases)
	- `score_u` is the predicted score for the positive instance
	- `score_v` is the predicted score for the negative instance
- notice how the larger the gap between `score_u` and `score_v` results in a lower loss!
- since the loss depends on two scores, it's used for [[learning to rank]]