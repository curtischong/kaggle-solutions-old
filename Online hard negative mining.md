https://paperswithcode.com/method/ohem
- Some object detection datasets contain an overwhelming number of easy examples and a small number of hard examples.
- OHEM selects these hard examples automatically to improve training.
- It is a bootstrapping technique that modifies [SGD](https://paperswithcode.com/method/sgd) to sample from examples in a non-uniform way depending on the current loss of each example under consideration.