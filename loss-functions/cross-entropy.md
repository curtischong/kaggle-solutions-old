https://blmoistawinde.github.io/ml_equations_latex/#cross-entropy
measures the performance of a classification model whose output is a probability value between 0 and 1.

In binary classification (one target)

$$-{(y\log(p) + (1 - y)\log(1 - p))}$$
In multiclass classification (multiple targets: M > 2)
$$-\sum_{c=1}^My_{o,c}\log(p_{o,c})$$
M - number of classes
log - the natural log
y - binary indicator (0 or 1) if class label c is the correct classification for observation o
p - predicted probability observation o is of class c


https://www.quora.com/What-are-some-advantages-and-disadvantages-of-using-cross-entropy-as-an-error-measure-when-training-neural-networks
#### pros
- It penalizes the model more strongly for incorrect predictions that are confident, which can lead to better-calibrated models that are less overconfident in their predictions.
#### cons
- Is sensitive to class imbalance, will be biased to the majority class
- Is sensitive to outliers and noise