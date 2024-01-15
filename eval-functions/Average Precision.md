https://www.kaggle.com/code/metric/event-detection-ap/notebook
- Use this metric when you want to see how well your model identifies (not predits!) "events" in a time series
- Here's how it works:
	1) Predicted events are matched with ground-truth events.
	2) Each group of predictions is scored against its corresponding
	    group of ground-truth events via Average Precision
	3) The multiple AP scores are averaged to produce a single
	    overall score.

Note: This metric could be abused:
![[Child Mind Institute - Detect Sleep States#^ep8pd9]]
 - you can abuse this to get a better score












