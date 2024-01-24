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
- A common approach is to treat it as a binary classification problem, cause the better you can predict each row's score, the more accurate it's rank will be
##### Summary

- okokoko There were A LOT OF THE TOP KAGGLE GRANDMASTERS IN THIS COMPETITION. just gushing sry
##### Solutions
(1st)
- https://www.kaggle.com/c/google-quest-challenge/discussion/129840
	- [[use intermediate layer results (weighted)]]
	- Using CLS outputs from all BERT layers rather than using only the last one. We did this by using a weighted sum of these outputs, where the weights were learnable and constrained to be positive and sum to 1
- some grandmasters are talking in the comments:
	- they didn't use thresholding, instead they placed each prediction into buckets (to generate the final result? not sure)
		- "But the threshold does a very similar thing. I actually tried both approaches on a very robust CV setup and fitting to distribution was always way worse for us. This was also confirmed when using rank-based approaches which inherenctly care more about the distribution."
		- "ANyway, binning and thresholding are quite similar"
- "Well, there is some post-processing. But tuning thresholds for each target was clearly seen as a straight way to overfitting, especially with 30 targets and so small public test size."
- "This model is initialized with original BERT weights, then finetuning with SX data, finally reusing the checkpoint in our pipeline. A subtle intermediate step was to delete classifier weights and biases from the SX checkpoint, because we had to switch from 6 SX targets to 30."
	- not sure if this is a trick or not. prob not
(2nd) - Very robust CV. Very nicely formatted target
- https://www.kaggle.com/competitions/google-quest-challenge/discussion/129978
	- ### Modifying the metric
		- Given the metric is rank-based, and given targets are not binary, it seemed important to be able to predict values that are neither 0 or 1 correctly.
			- they tried to use one-hot encoding of the targets (since some targets had a small number of distinct values)
				- For example: the "answer_type_procedure" column only had this as targets:
					- 0.000000, 0.333333, 0.666667, 1
					- more than one row can have these targets (cause it was graded via a rubric, not relatively between rows)
				- However, it didn't work, cause [[one-hot encoding]] destroys the ordering of targets
					- maybe [[thermometer encoding]] would've worked?
	- ### Cross validation
		- using ONLY [[GroupKFold]] (where the examples are grouped by question body) didn't work
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
			- Ignore spelling column (labels weren't very precise). [[drop bad targets from CV]]
			- Final CV is a mean of 5 folds.
	- [[differential learning rate]]
	- #### Training models
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