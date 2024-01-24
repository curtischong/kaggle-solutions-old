- When you take your trained model and label new data with your model's predictions
- After you have the new predictions, concat them with your old training data and retrain everything. This typically yields better results

### Important Notes:
- When using pseudo-labeled data, you want to make sure that you are cross validating properly:
	- https://www.kaggle.com/c/google-quest-challenge/discussion/129840
	- The situation:
		- you want to see if adding new labelled data from this data source helps your CV
	- 1) Assume you trained a model **on all** of your labelled training data
	- 2) Then you label some data
	- 3) You train your model on the old training data + the new pseudo-labelled data you made
	- 4) Now you do CV on your new model to see if the CV improves
	- **THERE IS IMPLICIT DATA LEAKAGE**
		- Why? Cause when you labelled the new dataset, the model was trained on all the data
		- so when you do your CV, the pseudo-labeled training data is DERIVED from information on your TEST fold
	- "In order to fix the problem, we generated 5 different sets of pseudolabels where for each train/val split we used only those models that were trained using only the current train set."
		- your pseudolabels will be less accurate, but that's fine. you're just doing CV
	- Before you submit, just retrain your model on all data before doing pseudo-labelling.