**Link:** https://www.kaggle.com/c/tweet-sentiment-extraction/overview
**Problem Type:** [[substring segmentation]]
**Input:** 
- a tweet
- each tweet is classified as either neutral, positive, or negative
	- I assume a human labels the sentiment
**Output:** the substring (selected_text column in train.csv) that determines the sentiment of the model
**Eval Metric:** [[jaccard similarity (aka Intersection Over Union)]] between your output and the selected_text column
##### Summary

**Data quality issues:**
- https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159254
- when humans tweet, they might add extra spaces:
	- "is back home now      gonna miss every one"
- so before they gave the tweets to the annotators, they remove consecutive spaces from the original tweet:
- ![[Pasted image 20240123115608.png]]
- since the label is retrieved on the original text, the dataset's target labels are WRONG (if there were consecutive spaces before the labelled text)
- Note: this isn't the only source of data quality issues. more cleaning was required
	- "What about when the selected_text should be "happy" but we find" happy fo"? did you find an explanation for these examples?"
		- ans: "no"
	- https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159254#889434
##### Solutions
- (1st) No post processing needed - use a character-level model instead
	- brief solution: https://www.kaggle.com/competitions/tweet-sentiment-extraction/discussion/159264
		- notebook for training character models (magic - fixes offset): https://www.kaggle.com/code/theoviel/character-level-model-magic/notebook
	- https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159477
		- how they did the concatenation: https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159254
	- heartkilla's models
		- custom labels: [[jaccard similarity (aka Intersection Over Union)]]-based Soft Labels [[custom loss]]
			- why? Cause Cross Entropy doesn’t optimize Jaccard directly
			- SoftIOU "soft intersection over union" didn't help
		- once he got his new soft labels, he was optimizing kldivergence 
		- his [[RoBERTa]] model's goal is for each of the output tokens, output the probability that the token is the start_idx or the end_idx
	- hikkii
		- [[sanity check]]: Traverse the top 20 of start_index and end_index, ensure start_index < end_index
			- cause the start_index and end_index are outputted 
	- theo's models
		- [[Multi Sample Dropout]]
		- [[Sequence bucketing]]
	- clev's models
		- used [[label smoothing]] 
		- [[Multi Sample Dropout]]
		- Discriminative learning
			- I'm not sure what this is
	- how did they fix the offset issue?
		- rather than doing post-processing, they **used a second layer model** to predict the correct offsets (this is their magic)
			- these are their character embedding models
	- [[pseudo-labeling]] 
- (3rd) No postprocessing. Their models just learned the noise
	- https://www.kaggle.com/competitions/tweet-sentiment-extraction/discussion/159269
	- solution code: https://github.com/suicao/tweet-extraction/
	- the main idea: "Basically you predict the start index, get the _k_ hidden states at the _top-k_ indices. for each hidden state, concat it to the end index logits and predict the corresponding _top-k_ end indices."
	- [[thefuzz (prev fuzzywuzzy)]]
		- I think he used it to fuzzy match the output substring with the actual text in the tweet in case the output wasn't exactly the same as the tweet
			- https://github.com/suicao/tweet-extraction/blob/f8207b61cc4f87b58a09e1362875c1a750842c86/utils.py#L29
			- ok yeah, he just returns the substring that has the highest match with the generated output
				- note the generated output is just a substring output (via indices, not GPT-like generation
	- "I used the fastai style freeze-unfreeze scheme since the head is quite complicated."
	- looking at their code:
		- position_outputs[idx,:] = [start_idx + len(input_ids_0) + 2, end_idx + len(input_ids_0) + 2]
- (4th)
	- https://www.kaggle.com/competitions/tweet-sentiment-extraction/discussion/159499
		- he also has 4 model heads:
			- 1) QA dense head (just a linear layer without any dropout) for predicting start and end tokens
				- for each token in the output, predict if it's the start token, or if it's the end token
				- "Loss is computed with [[KLDivergenceLoss]] to add [[label smoothing]]":
					- "true target token is given 0.9 probability and two of its neighbors (left and right) both take 0.05"
					- I think they're saying that kldivergence basically adds label smoothing
			- 2) Linear layer to predict binary target for each token: if it should be in selected text or not.
				- very similar to 1) but now, every token in the answer is given a probability of 1
			- [[alternative targets]]
			- 3) linear layer to predict a sentiment of each token.
			- 4) Two linear layers with ReLU in between to predict the sentiment of the whole tweet.
		- ### _Inference phase_
			- [[derive results from logits]]
				- Using 1st model head (the one that predicts the probability of each token being a start OR end token)
					- they took the logits (i.e. the signal right before it's squashed between 0-1 via a sigmoid)
					- and summed up all n choose 2 logit values
					- then they just selected the top 3 of these three values to get the top position pairs
						- these pairs are the most likely spots of being the start and end indexes!
	
##### Important notebooks/discussions

#### Takeaways
- nobody used a regression strategy to predict the start/end indices
	- they don't "use bert to output 2 integers from 1 to n representing the start and end idx of the substring"
	- for each token in the input, your model predicts the probability that it's in the tweet substring or not
		- so if there's n tokens as the input, the output is dimension n
- you can take the logits of the final layer and do things with it... TODO fully understand
- 