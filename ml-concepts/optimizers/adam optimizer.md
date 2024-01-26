- optimizes the learning rate for each parameter in your model
- Kaparthy likes to use Adam with a learning rate of [3e-4](https://twitter.com/karpathy/status/801621764144971776?lang=en) (to set baselines)
	- "In my experience Adam is much more forgiving to hyperparameters, including a bad learning rate."

- Q: Doesn't Adam set the learning rate for me? Why do I have to set one?
	- The value you provide is the initial ballpark that Adam uses (adam can then decide what learning rate to use later)