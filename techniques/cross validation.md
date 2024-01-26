https://medium.com/geekculture/cross-validation-techniques-33d389897878
- there's lots of diff cv methods here. but I'm not sure which ones are used.
- I'm only including ones I've seen used

- [[GroupKFold]]
- [[forward chaining cross validation]]
- [[Blocked Cross-Validation]]

### How to know if your CV is bad

- https://www.kaggle.com/competitions/g-research-crypto-forecasting/discussion/323098
	 - An easy way to see if your CV scores have too much variance is to look at a plot of CV score vs. a parameter you're tuning. A good plot will usually be smooth, with a knee and a plateau, or maybe a peak or a valley. If the plot looks too noisy to clearly see those things, then try it again, with different seeds, and see if you get different results. If you do, then your CV results are too noisy. You can try using more folds, or running CV several times with different random seeds and averaging.)


#### best way to cross validation over time
- [[forward chaining cross validation]] is pretty popular
