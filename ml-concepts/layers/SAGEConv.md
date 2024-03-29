- It is the conv layer that [[GraphSAGE]] uses.
- How it works:
	- 1) After the node gets the features from the neighbours, it aggregates them (via any of: mean, svg, max, etc.)
	- 2) Next, the layer concats the **aggregated neighbour features** with **it's own features**
		- `CONCAT(node_features, aggregated_neighbour_features)`
	- 3) after we have our new feature set, we apply BatchNorm or LayerNorm
- Notice how this is a convolutional layer, so we can apply all the techniques we use in conv nets ONTOP of graph neural nets