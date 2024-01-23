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
(1st) No post processing needed
- https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159477
	- how they did the concatenation: https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159254
- custom loss: [[jaccard similarity (aka Intersection Over Union)]]-based Soft Labels [[custom loss]]
	- why? Cause Cross Entropy doesn’t optimize Jaccard directly
	- SoftIOU "soft intersection over union" didn't help
	- 
- [[pseudo-labeling]]

##### Important notebooks/discussions

#### Takeaways