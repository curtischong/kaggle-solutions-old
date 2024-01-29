https://www.kaggle.com/code/carlmcbrideellis/what-is-adversarial-validation
- let us know if our test data and our training data are similar
- we combine our `train` and `test` data, labeling them with say a `0` for the training data and a `1` for the test data, mix them up, then see if we are able to correctly re-identify them using a binary classifier.
- If we cannot correctly classify them, _i.e._ we obtain an area under the [[receiver operating characteristic curve (ROC)]] of 0.5 then they are indistinguishable and we are good to go.