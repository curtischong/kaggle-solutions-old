**Link:** https://www.kaggle.com/competitions/google-quest-challenge
**Problem Type:** [[learning to rank]]
**Input:** each row contains a question, and the answer to it (on a Q&A site)
- Note: multiple rows can come from the same question, we group by `question_body`
	- but they are in different rows since there were more than one answer to the question
**Output:** 
**Eval Metric:** [[Spearman's correlation Coefficient]]
- The goal is to see **how well your model can rank** Q&A forum posts based on these target columns:
	- question_well_written
	- answer_helpful
	- answer_relevance
	- answer_type_reason_explanation
	- etc.
- there are 30 target columns
	- each target column is a value from [0,1]
	- for **each column**:
		- kaggle sorts your predictions (decreasing yhat)
			- the yhat you predict is just used to sort all the test rows
		- we now have an ordered array of [questionId, rank]
			- the rank is just the index of that questionId in the array
		- we now run spearman's correlation of this array AGAINST kaggle's rank of this array
	- since there are 30 columns, kaggle does it 30 times^
	- The final score is the average of all 30 of these spearman correlations
- A common approach was to treat it as a binary classification problem, cause the better you can predict each row's score, the more accurate it's rank will be
- IMPORATANT NOTE:
	- these target variables can have duplicate values (since they were evaluated against a rubric, not against each other)
	- many of these targets only had 4 distinct values
		- 0, 1/3, 2/3, 1
	- so ppl in the competition talked about "thresholding" their predictions (and treating it as a classification problem, rather than a regression or even a ranking problem)
	- why does making it discrete matter? doesn't spearman corr only care about rank?
		- no. cause [[Spearman's correlation Coefficient]] REALLY CARES if two items have have same score vs if one is a bit higher than other
##### Summary
- There were A LOT OF THE TOP KAGGLE GRANDMASTERS IN THIS COMPETITION.
	- so the solutions are very high quality

##### Solutions
- (1st)
	- https://www.kaggle.com/c/google-quest-challenge/discussion/129840
		- solution code: https://github.com/oleg-yaroshevskiy/quest_qa_labeling
		- [[use intermediate layer results (weighted)]]
		- Using CLS outputs from all BERT layers rather than using only the last one. We did this by using a weighted sum of these outputs, where the weights were learnable and constrained to be positive and sum to 1
		- [[postprocess to match target distribution]]
			- https://github.com/oleg-yaroshevskiy/quest_qa_labeling/blob/730a9632314e54584f69f909d5e2ef74d843e02c/step11_final/blending_n_postprocessing.py#L55
				- code
					def postprocess_single(target, ref):
					    """
					    The idea here is to make the distribution of a particular predicted column
					    to match the correspoding distribution of the corresponding column in the
					    training dataset (called ref here)
					    """
					
					    ids = np.argsort(target)
					    counts = sorted(Counter(ref).items(), key=lambda s: s[0])
					    scores = np.zeros_like(target)
					
					    last_pos = 0
					    v = 0
					
					    for value, count in counts:
					        next_pos = last_pos + int(round(count / len(ref) * len(target)))
					        if next_pos == last_pos:
					            next_pos += 1
					
					        cond = ids[last_pos:next_pos]
					        scores[cond] = v
					        last_pos = next_pos
					        v += 1
					
					    return scores / scores.max()
		-  some grandmasters are talking in the comments:
		- they didn't use thresholding, instead they placed each prediction into buckets (to generate the final result? not sure)
			- "But the threshold does a very similar thing. I actually tried both approaches on a very robust CV setup and fitting to distribution was always way worse for us. This was also confirmed when using rank-based approaches which inherenctly care more about the distribution."
			- "ANyway, binning and thresholding are quite similar"
	- "Well, there is some post-processing. But tuning thresholds for each target was clearly seen as a straight way to overfitting, especially with 30 targets and so small public test size."
	- "This model is initialized with original BERT weights, then finetuning with SX data, finally reusing the checkpoint in our pipeline. A subtle intermediate step was to delete classifier weights and biases from the SX checkpoint, because we had to switch from 6 SX targets to 30."
		- not sure if this is a trick or not. prob not
- (2nd) - Very robust CV. Very nicely formatted target
	- https://www.kaggle.com/competitions/google-quest-challenge/discussion/129978
	- ### Modifying the metric
		- Given the metric is rank-based, and given targets are not binary, it seemed important to be able to predict values that are neither 0 or 1 correctly.
			- they tried to use one-hot encoding of the targets (since some targets had a small number of distinct values)
				- For example: the "answer_type_procedure" column only had this as targets:
					- 0.000000, 0.333333, 0.666667, 1
					- more than one row can have these targets (cause it was graded via a rubric, not relatively between rows)
				- However, it didn't work, cause [[one-hot encoding]] destroys the ordering of targets
					- maybe [[thermometer encoding]] would've worked?
		- The metric they settled on:
			- [[binary encoded categorical ordinal targets]]

	- ### Cross validation
		- using ONLY [[GroupKFold]] (where the examples are grouped by question body) **didn't work**
			- 1) "Test data is different to training data in the sense that it only has one question-answer pair sampled out of a group of questions in train data. There can be stark noise for labels of the same question, which is why this needs to be addressed robustly."
				- I don't 100% know what this means
			- 2) cause there are a few columns with very rare events and also a lot of noise within those rare events
		- their CV solution:
			- 1) in each validation fold, first groupby question_body
				- out of all your unique question_body, randomly pick 100 of them
					- and ONLY PICK ONE ANSWER to validate against
			- 2) your model now has 100 predictions
			- 3) Calculate the median score across these 100 samples and report.
				- how is this score calculated? you can't use spearman with one value
				- maybe they just look at [[MAELoss]] between the yhat and y for that one row
			- Ignore spelling column (labels weren't very precise). [[drop bad targets from CV]]
			- Final CV is a mean of 5 folds.
	- post processing
		- [[clip outputs to be within range]]
			if their final predictions are `x=[0.01, 0.015, 0.02, 0.03, 0.04]` and our optimal thresholds are `coefs=[0.016, 0.029]`
			- then we would clip the data with `np.clip(x, coefs[0], coefs[1])`
			- leading to `x=[0.016, 0.016, 0.02, 0.029, 0.029]` effectively generating ties at the edges.
		- how did they figure out good clips?
			- Sample 1000 times single question-answer pairs from multiple questions
			- Generate thresholds that optimize the median score of all 1000 samples
	- #### Training models
		- used [[differential learning rate]]
		- All “normal” tricks that worked in past computer vision or other NLP competitions (e.g. concat of Max and Mean pooling) did not improve cv. Also using a sliding window approach to capture more text did not work.
			- they tried sliding window probably because transformers at the time had limited context, you had to remove the middle and keep the ends
		- saw a big jump when they tried to freeze the transformers and only train the model head for 1 epoch, before fine-tuning
			- isn't fine-tuning just more training?
				- ahhh, by fine-tuning they probably just stop freezing the transformers
		- We further improved than this 2 step approach by using different learning rates for transformer and head, together with a warm up schedule, which enabled us to get rid of the freezing step in general.
		- [[use intermediate layer results (weighted)]]
		- their "siamese model"
			- normally, siamese is used to determine if two ppl's faces look similar for facial recognition
			- but they say "siamese model" just because they are referencing: "feed question and answer twice through the same transformer"
			- then they take the final embeddings and concatenate the featrues before feeding it into a dense layer with 30 "target" values
				- target is in ""s cause: "(For the sake of simplicity we show the architecture for the original 30 targets. It was adapted to work with the binarized targets.)"

##### Important notebooks/discussions
- https://www.kaggle.com/code/carlolepelaars/understanding-the-metric-spearman-s-rho/notebook
	- many ppl were just using binary_crossentropy since the targets "look" like a binary classification problem

#### Takeaways