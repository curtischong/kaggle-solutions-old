https://www.kaggle.com/code/carlmcbrideellis/what-is-adversarial-validation
- let us know if our test data and our training data are similar
- we combine our `train` and `test` data, labeling them with say a `0` for the training data and a `1` for the test data, mix them up, then see if we are able to correctly re-identify them using a binary classifier.
- If we cannot correctly classify them, _i.e._ we obtain an area under the [[receiver operating characteristic curve (ROC)]] of 0.5 then they are indistinguishable and we are good to go.
	- "Ideal is 0.5. Which is what you'd like to get. Certainly nothing beyond 0.49-0.51 range."
	- https://www.kaggle.com/code/tunguz/lanl-adversarial-validation-shakeup-is-coming/comments#537857
- you should probably use this technique as a [[sanity check]]