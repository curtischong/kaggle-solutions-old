- Sometimes, passing in dimensions from one part of the network to another could lead to many more params
	- e.g. taking in this input:
		- 5 x 3 x 20
	- is smaller than taking in this input:
		- 15 x 20
	- here we split the 15 into 5 and 3
	- we call the "3" dimension the channels
- note: it takes trial and error to figure out how many channels to split a tensor into
- see https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447449 for an example