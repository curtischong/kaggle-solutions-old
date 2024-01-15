Link: 
Problem Type: 
Input: 
Output: 
Eval Metric: [[RMSE]]
##### Summary
Given only keystroke information, predict a student's essay score [0, 6], in increments of 0.5.
- all alphanumeric characters were replaced with `q` so you couldn't decipher the original essay text
##### Solutions

- (3rd) Dieter's team.
	- https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/discussion/466906
		- 8fold cross validation
		- [[identify domain shift]]: Deberta worked better on CV, while GBM were relatively bad on CV, but good on LB.
			- That indicates a slight domain shift between LB and train set.
			- Our guess is the essays might be split by topic or student year.
		- DeBERTa
			- replaced the obscurification character `q` with `i` or `X` when training debertas on the reconstructed essa text
				- prob cause the deBertas have these tokenizations in them:  i, ii, iii, or X, XX, XXX
				- [[Training own Tokenizer]] noticing that these specialized tokenizations led to better results, they trained their own tokenizer specifically for X, XX, or XXX.
					- This led to much better results
			- **External Data Source:** obfuscated the persuade corpus in a similar manner as the train dataset (for more data)
				- They [[Train on external data first]]
					- so DeBERTa they saw the competition data, it already knew how obfuscated text looked like
				- [[Masked Language Modeling (MLM)]]
			- used a [[Squeezeformer layer]] to derive semantic features
			- added three features:
				-  cursor position, etc
				- Adding further keystroke features did not help, and we needed heavy dropout/augmentation on the three we added.
			- The final components were as following
				- deberta-v3-base trained with q replaced by i
				- deberta-v3-large trained with q replaced by i (with first 12 layers frozen in finetuning, to avoid overfit) [[Freezing Layers]]
				- deberta-v3-base trained with q replaced by X
				- deberta-v3-base trained with custom spm tokenizer
			- "we used positive Ridge Regression on Out-of-fold (OOF) predictions to determine blending weights."
				- What does this mean? Does it mean that they took yhat to predict y using ridge regression and look at the error?
					- then using this error, determine the blend weights?
		- ##### Postprocessing
			- For some models we clipped predictions at [0.5,6.] but it did not really make a difference.
		- ##### What did not help
			- Using the deleted text
			- Stacking
			- Adding more keystroke features to the deberta based model.
			- More squeezeformer layers

##### Important notebooks/discussions

#### Takeaways