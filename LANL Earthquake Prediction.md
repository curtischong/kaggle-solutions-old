**Link:** https://www.kaggle.com/c/LANL-Earthquake-Prediction
**Problem Type:** [[regression]]
**Input:** the acoustic data (the shaky line that is draw by seismometers)
**Output:** the time (in seconds) until the next laboratory earthquake
**Eval Metric:** [[Mean absolute error (MAE)]]
##### Summary
- you’ll predict the time remaining before laboratory earthquakes occur from real-time seismic data.
- these aren't real earthquakes. it's a lab setup:
	- ![[Pasted image 20240129154843.png|400]]
- **Why did many ppl overfit the public lb?**
	- 
##### Solutions
- (1st) PSI's solution
	- "After doing abovementioned signal manipulation, we had more trust in our calculated features and could focus on better studying differences between train and test data feature distributions"
	- we calculated a handful of features for train and test and tried to find a good subset of full earth-quakes in train, so that the overall feature distributions are similar to those of the full test data.
	- a simple shuffled 3-[[kfold]]
	- [[adversarial validation]] shows that the signal had a certain time-trend that made mean or quantile features unreliable
		- e.g. mean number of "spikes" in a sliding window of 100ms of data
	- One of our best final LGB model only used four features:
		- (i) number of peaks of at least support 2 on the denoised signal
		- (ii) 20% percentile on std of rolling window of size 50
		- (iii and iv) 4th and 18th coefficient in the series of [[Mel frequency cepstral coefficients (MFCC)]] 
	- Those 4 are decently uncorrelated between themselves, and add good diversity.
	- For each feature we always only considered it if it has a p-value >0.05 on a KS statistic of train vs test. [[considering features]]
	- **why does adding noise then subtracting the median de-noise the signal?**
		- asdasd
		- Our features are then calculated on this manipulated signal.
	- [[KS statistic]] showed that there's a drift between the train and test data
		- so we decided to sample the train data to make it look more like we expect test data to look like (only from looking at feature distributions)
		- we calculated a handful of features for train and test and tried to find a good subset of full earth-quakes in train, so that the overall feature distributions are similar to those of the full test data.
		- We did this by sampling 10 full earthquakes multiple times (up to 10k times) on train, and comparing the average KS statistic of all selected features on the sampled earthquakes to the feature dists in full test.
			- **what does sampling 10 full earthquakes multiple times mean?**
		- "The x-axis is the average target of the selected EQs in train and the y-axis is the KS statistic on a bunch of features comparing the distribution of that feature for the selected EQs vs the full test data. We can see that the best average KS-statistic is somewhere in the range of 6.2-6.5. You can also see nicely here that a problematic feature like the green one deviates clearly from the rest, this would be a feature we would not select in the end."
##### Important notebooks/discussions

#### Takeaways