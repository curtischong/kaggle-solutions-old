https://www.kaggle.com/code/shonenkov/wbf-approach-for-ensemble
- Basically, when we have lots of bounding box predictions in image identifying tasks, WBF will:
	- 1) Look at all the boxes
	- 2) Output a a new box that is the weighted average of all the candidate boxes
- This is better than the NMS approach, which simply picks one box (and removes the other boxes)
