- This is a really important paper that not only introduces GPS, but explains (in a very simple manner) embedding techniques GNNs have used in the past
https://openreview.net/pdf?id=lMMaNf6oxKM
- There have been many recent works on PE and SE, notably on Laplacian PE [14, 36, 3, 57, 39], shortest-path-distance [38, 63], node degree centrality [63], kernel distance [ 44 ], random-walk SE [16 ], structure-aware [9, 6 , 5], and more. Some works also propose dedicated networks to learn the PE/SE from an initial encoding [ 36, 16 , 39 , 9]. 
- [[graph laplancian]]
- Positional encodings (PE) are meant to provide an idea of the position in space of a given node within the graph. Hence, when two nodes are close to each other within a graph or subgraph, their PE should also be close. **A common approach is to compute the pair-wise distance between each pairs of nodes or their eigenvectors**
- 