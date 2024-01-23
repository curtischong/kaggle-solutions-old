use case:
- you are training a model to detect cancer cells
	- but you only give it images of cancer cells that came from one patient
- If there are MORE images from that patient in the validation set, your CV is biased
	- cause your model may only learn **how cancer looks for that one patient**
Solution
- Make sure all the data points for one person is in the train set OR in the test set
	- only have Bob / Tracy / Mary in the train set
	- only have Dillian's images in the test set
- **DO NOT MIX BOB'S IMAGES IN THE TRAIN AND TEST SET**

- GroupKFold prevents this mixing from happening