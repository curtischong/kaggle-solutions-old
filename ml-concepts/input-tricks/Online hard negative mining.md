- A hard negative is **an example that is falsely and confidently classified as positive by a model**
https://paperswithcode.com/method/ohem
- Some object detection datasets contain an overwhelming number of easy examples and a small number of hard examples.
- OHEM selects these hard examples automatically to improve training.
- It is a bootstrapping technique that modifies [SGD](https://paperswithcode.com/method/sgd) to sample from examples in a non-uniform way depending on the current loss of each example under consideration.
- To do hard negative mining during training (making it online), the algorithm identifies examples that are past the margin of the model's boundary, meaning: it'll feed in examples that the model will classify wrong (so the model tries to use it to improve)
- implementation details the paper suggests:
	- 1) modify the loss layers to only backpropagate gradients for the global hard examples in the dataset
		- but this is inefficient, cause backpropagation already fixes hard examples for us. there's no point in first figuring out the global hard examples then ONLY backpropagating these specific examples
	- 2) use a secondary model that is readonly (so we only allocate memory for hte forward pass, and not the backward pass)
	- OK these implementations suck. I think the best way is to just **modify the loss function so that if the example is a hard negative example, we multiply the loss by a constant**