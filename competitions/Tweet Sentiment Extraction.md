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
		- custom loss: [[jaccard similarity (aka Intersection Over Union)]]-based Soft Labels [[custom loss]]
			- why? Cause Cross Entropy doesn’t optimize Jaccard directly
			- SoftIOU "soft intersection over union" didn't help
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

##### Important notebooks/discussions

#### Takeaways