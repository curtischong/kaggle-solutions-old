Link: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states
Problem Type: 
Input: wrist accelerometer data
Output: the times when a user wakes up AND the times when a user sleeps
Eval Metric: [[Average Precision]]
- https://www.kaggle.com/code/metric/event-detection-ap/notebook
- note: from (1st)'s solution, they say that:
	- The competition's evaluation metric doesn't differentiate predictions within a 30-second range from the ground truth event.
	- so your prediction has a margin of error of 30seconds and it'll be correct
##### Summary
Detect sleep onset and wake from wrist-worn accelerometer data


Note: if you submitted multiple predictions for the same sleep event, you'll get the same score (or better) ^ep8pd9
- https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/460516
- competitors abused this bias to get a better score
##### Solutions
- (1st)
	- https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459715
		- was based on this public notebook:
			- https://www.kaggle.com/code/danielphalen/cmss-grunet-train
				- [[KLDivergenceLoss]] was good because
					- 1) The target is "easily interpretable as a probability distribution"
					- 2) Ensembles are also easy to interpret, no matter how differently they are trained.
					- HOWEVER, he switched to [[BCEWithLogitsLoss]]
				- how to improve this notebook:
					- 1) use fastparquet
					- 2) add skip connections
		- For input scaling, SEModule was utilized. ([https://arxiv.org/abs/1709.01507](https://arxiv.org/abs/1709.01507))
			- SEModules
				- [Squeeze-and-Excitation block explained | by Taha Samavati | Medium](https://medium.com/@tahasamavati/squeeze-and-excitation-explained-387b5981f249)
				- basically: when we do a regular convolution, we apply the weights in the same way to all channels.
					- SE block is similar, but the importance of each channel is individually assessed based on its context
						- so it probably changes the kernel when doing the convolutions on a per-channel basis
		- As noted in several discussions and notebooks, there was a bias in the minute when ground truth events occurred.
		- A decaying target is created based on the distance from the ground truth event, with diminishing values as the distance increases.
			- ![[Pasted image 20240114143837.png]]
			- every epoch, he decays the target:
				- `targets = np.where(targets == 1.0, 1.0, (targets - (1.0 / config.n_epochs)).clip(min=0.0))`
				- this sharpens the peaks for detecting onsets / wakeups
					- ![[Pasted image 20240114150324.png]]
		- Missing data exists when the device is removed: https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459715
			- when that happens, they use rules to predict the events
		- Their post-processing improved their score by a lot
			- 1) understand that with this metric, if your prediction is within 30 seconds of a ground truth event, you get full points
			- 2) use two models:
				- model 1: every 5 seconds (the period of one time step), predict the probability of an onset / awake
				- model 2: only predict 1 if you think they woke up **on that second**. Otherwise, predict 0
					- This predicts a binary flag indicating whether the ground truth event exists at the point.
			- They now use a greedy algorithm to find the points where the sum of the scores of its neighbours are the best
				- If they already "used up" score from one window, they cannot reuse it for another
			- "The reason we are only calculating for two points within a minute is to reduce computational complexity."
			- "In our case, performance did not improve with post-processing alone. (w/o daily normalization of the score)
				- "The 2nd level model and daily normalization of the score were crucial elements in improving performance."
	- https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/441470
	- [pure heuristic approach (HDCZA)](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/453267)
		- notebook: https://www.kaggle.com/code/tatamikenn/sleep-hdcza-a-pure-heuristic-approach-lb-0-447?scriptVersionId=149818796
		- "Since ~37% of nights in training sequences has no sleep logs, I believe training model of predicting existence of sleep logs in the target night will get higher score in LB."
		- it was based on this paper: https://www.nature.com/articles/s41598-018-31266-z#data-availability
			- the implementation details start at the "Accelerometer data preparation" section
			- Heuristic algorithm looking at Distribution of Change in Z-Angle (HDCZA)
			- Main steps of the heuristic (to find the SPT-window (The Sleep Period Time window)):
			- ![[Pasted image 20240114142212.png]]
		- angleZ seems like an important feature!
- (2nd)
	- https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459627
	- used [[WBF ensembling]] rather than averaging for the final ensemble prediction
	- After getting the score for each step, he uses a LGBM model to predict the scores for the steps (but the steps are shifted by a bit)
		- this is so he can get more predictions that are nearby. the extra predictions won't harm his score
	- he concats these new predictions back onto the original table from step 2
- (3rd)
	- https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459599
	- feature engineering
		- only had 3 features. Any more features was innacurate
		- Made anglez absolute, giving +0.002 on local validation.
		- For the only two variables we had (anglez and enmo), we tried to find useful aggregations(diff, mean, median, skew, etc…), but the only thing that seemed to work was the standard deviation (**anglez_abs_std** and **enmo_std**).
		- Detecting noise: We realized that when exactly the same value is repeated in the same series at the same hour, minute and second, this was basically noise.
		- To incorporate temporal information into the model, we decided to add 2 frequency encoding variables (one for onsets and one for wakeups) at the hour-minute level.
#### Takeaways