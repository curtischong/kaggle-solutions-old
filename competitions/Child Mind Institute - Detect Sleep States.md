Link: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states
Problem Type: 
Input: wrist accelerometer data
Output: the times when a user wakes up AND the times when a user sleeps
Eval Metric: [[Average Precision]]
- https://www.kaggle.com/code/metric/event-detection-ap/notebook
##### Summary
Detect sleep onset and wake from wrist-worn accelerometer data


Note: if you submitted multiple predictions for the same sleep event, you'll get the same score (or better)
- https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/460516
- competitors abused this bias to get a better score ^ydz6b7
##### Solution Links
(1st)
- https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459715
	- was based on this public notebook:
		- https://www.kaggle.com/code/danielphalen/cmss-grunet-train
			- uses [[KLDivergenceLoss]] because
				- 1) The target is "easily interpretable as a probability distribution"
				- 2) Ensembles are also easy to interpret, no matter how differently they are trained.
			- how to improve this notebook:
				- 1) use fastparquet
				- 2) add skip connections
	- For input scaling, SEModule was utilized. ([https://arxiv.org/abs/1709.01507](https://arxiv.org/abs/1709.01507))
		- SEModules
			- [Squeeze-and-Excitation block explained | by Taha Samavati | Medium](https://medium.com/@tahasamavati/squeeze-and-excitation-explained-387b5981f249)
			- basically: when we do a regular convolution, we apply the weights in the same way to all channels.
				- SE block is similar, but the importance of each channel is individually assessed based on its context
					- so it probably changes the kernel when doing the convolutions on a per-channel basis

### Important kernels
- https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/441470
- [pure heuristic approach (HDCZA)](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/453267)
	- notebook: https://www.kaggle.com/code/tatamikenn/sleep-hdcza-a-pure-heuristic-approach-lb-0-447?scriptVersionId=149818796
	- "Since ~37% of nights in training sequences has no sleep logs, I believe training model of predicting existence of sleep logs in the target night will get higher score in LB."
	- it was based on this paper: https://www.nature.com/articles/s41598-018-31266-z#data-availability
		- the implementation details start at the "Accelerometer data preparation" section
		- Heuristic algorithm looking at Distribution of Change in Z-Angle (HDCZA)
		- Main steps of the heuristic (to find the SPT-window (The Sleep Period Time window)):
		- ![[Pasted image 20240114142212.png]]

#### Takeaways