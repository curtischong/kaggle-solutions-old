**Link:** https://www.kaggle.com/c/bengaliai-speech/overview
**Problem Type:** [[transcription]]
**Input:** mp3 files
**Output:** the transcribed Bengali text
**Eval Metric:** [[Word Error Rate]]
##### Summary

Bengali dialects (especially those spoken by Muslim religious sermons) aren't transcribed well.
- There are no robust open-source speech recognition models for Bengali
- (the Google Speech API for Bengali has a Word Error Rate of 74% for Bengali religious sermons).
##### Solutions
- (1st) - openai whisper, custom whisper tokenizer
	- cool tricks
		- [[Spectrogram dithering]]
		- Resampling 16khz->8khz->16khz as augmentation [[downscale upscale examples]]
	- they realized that whisper's tokenizer was slow on Bengali audios [[identifying slow feature generation]]
		- so they retrained it using a 12k vocabulary on Bengali texts.
		- "How did you initialize the word embeddings after changing the tokenizer?"
			- they probably didn't and just finetuned the model on those new tokens
				- I guess it's hard to set initial word embeddings (so your model converges faster - see kaparthy's tips for ml) after working in such high dimentional space
			- The original Whisper tokenizer employs character-level tokens for low-resource languages, which can be time-consuming. So we trained a [[BPE tokenizer]] with 12,000 tokens specifically for Bengali text. We then replaced some tokens in the Whisper tokenizer with these. We carefully replaced tokens after the 10,000th position of the original Whisper's tokens.
				- so their tokenizer has a vocab of 22,000 tokens?
					- I guess they removed it after the 10,000th position since the tokens past those are super rare?
			- I should look into how to modify tokenizers. t
	- augmented the speech using Libsonic
	- downloaded many youtube videos and pseudolabelled them
		- I think he used a previous version of his model to do the labelling for hte next version of his model
		- How did you handle the alignment between audio and transcription?
			- Long youtube audios are splitted using with VAD. Reject too short or too long audio segments and let's say use only segments between 5 and 22 seconds.
		- used webrtcvad for the [[voice activity detection (VAD)]] library
	- many ppl didn't succeed finetuning with whisper cause they didn't remove wrong annotations from the dataset
		- "Because the competition dataset was not validated, the initial model was trained on OpenSLR datasets"
			- yeah they really didn't trust the trainin data, which is why their score was so much better than everyone else's

##### Important notebooks/discussions
- explaining the competition
	- https://www.kaggle.com/code/sujaykapadnis/bengali-speech-recognition-for-everyone
	- the test set has 18 different dialects
		- the train set only has 1 dialect
	- each training sample is either a "train or valid" sample
		- valid samples have been manually reviewed and corrected
		- train samples have only been algorithmically labelled
		- both are **drawn from the same distribution**.
#### Takeaways
- use good training data.  if you are given bad training data, don't make a cake out of garbage. find better ingredients!