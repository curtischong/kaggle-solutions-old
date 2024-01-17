 https://dongreanay.medium.com/pre-training-llms-techniques-and-objectives-a75a1bf274b2#:~:text=The%20MLM%20objective%20involves%20randomly,by%20the%20non%2Dmasked%20tokens.
 - You feed a sentence into a language model. But you mask a few tokens in the sentence. The model is then supposed to fill in the missing token
 - The error metric is [[eval-functions/cross-entropy loss]]
	 - cause the language model predicts a probability for every token in its vocabulary, yhat is the probability that the model selected the masked token
	 - so even if the most probable token is not the correct token, it doesn't matter. since we only look at the probability of the masked token being selected
 - Note: there may be other error metrics (maybe the semantic similarity of the predicted token with the actual masked token?)