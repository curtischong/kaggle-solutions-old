
- clustering algo where each point belongs in only 1 cluster
- clusters are always circular
- steps
- 1) select a number of classes/groups to use and randomly initialize their respective center points
- 2) classify each datapoint as the cluster with the nearest group cluster.
    - nearest depends on a [[distance metrics]]
- 3) recompute the group center by taking the mean of all vectors in the group
- 4) repeat 2 & 3 until convergence